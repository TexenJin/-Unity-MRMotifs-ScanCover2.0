using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using UnityEngine;

/// <summary>
/// Deterministic V2 input tape. It stores the exact 32-bit NDC depth values
/// read from the callback-owned stereo packet together with that packet's
/// projection/view matrices. Replay performs a deterministic CPU unprojection
/// from those inputs; it does not claim bit identity with the live GPU's
/// world-position arithmetic. File I/O is isolated on one bounded writer
/// thread; when storage cannot keep up, frames are dropped instead of
/// stalling reconstruction or silently changing the depth representation.
/// </summary>
public static class RoomSetupReplicaV2Tape
{
    private const uint Magic = 0x32565352u; // RSV2
    private const uint FrameMarker = 0x4d415246u; // FRAM
    private const uint EndMarker = 0x21444e45u; // END!
    private const int Version = 1;

    public sealed class Frame
    {
        public int platformFrame;
        public double captureTime;
        public Matrix4x4 leftWorldToClip;
        public Matrix4x4 rightWorldToClip;
        public Matrix4x4 leftProjectionInverse;
        public Matrix4x4 rightProjectionInverse;
        public Matrix4x4 leftViewInverse;
        public Matrix4x4 rightViewInverse;
        public float[] leftNdc;
        public float[] rightNdc;
    }

    public sealed class Writer : IDisposable
    {
        private readonly BlockingCollection<Frame> _queue =
            new BlockingCollection<Frame>(3);
        private readonly Thread _thread;
        private readonly BinaryWriter _writer;
        private readonly int _sampleCount;
        private readonly int _maximumFrames;
        private int _acceptedFrames;
        private int _writtenFrames;
        private int _droppedFrames;
        private Exception _failure;
        private bool _disposed;

        public string Path { get; }
        public int AcceptedFrames => Volatile.Read(ref _acceptedFrames);
        public int WrittenFrames => Volatile.Read(ref _writtenFrames);
        public int DroppedFrames => Volatile.Read(ref _droppedFrames);
        public Exception Failure => _failure;

        public Writer(
            string path, int width, int height, int maximumFrames)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException(nameof(width));
            Path = path;
            _sampleCount = checked(width * height);
            _maximumFrames = Mathf.Max(1, maximumFrames);
            string directory = System.IO.Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            _writer = new BinaryWriter(new FileStream(
                path, FileMode.Create, FileAccess.Write, FileShare.Read,
                1 << 20, FileOptions.SequentialScan));
            _writer.Write(Magic);
            _writer.Write(Version);
            _writer.Write(width);
            _writer.Write(height);
            _thread = new Thread(WriteLoop)
            {
                IsBackground = true,
                Name = "RoomSetupReplicaV2TapeWriter"
            };
            _thread.Start();
        }

        public bool TryEnqueue(
            int platformFrame,
            double captureTime,
            Matrix4x4 leftWorldToClip,
            Matrix4x4 rightWorldToClip,
            Matrix4x4 leftProjectionInverse,
            Matrix4x4 rightProjectionInverse,
            Matrix4x4 leftViewInverse,
            Matrix4x4 rightViewInverse,
            float[] leftNdc,
            float[] rightNdc)
        {
            if (_disposed || _failure != null ||
                leftNdc == null || rightNdc == null ||
                leftNdc.Length < _sampleCount ||
                rightNdc.Length < _sampleCount)
                return false;
            int slot = Interlocked.Increment(ref _acceptedFrames);
            if (slot > _maximumFrames)
            {
                Interlocked.Decrement(ref _acceptedFrames);
                return false;
            }

            float[] left = ArrayPool<float>.Shared.Rent(_sampleCount);
            float[] right = ArrayPool<float>.Shared.Rent(_sampleCount);
            Array.Copy(leftNdc, left, _sampleCount);
            Array.Copy(rightNdc, right, _sampleCount);
            var frame = new Frame
            {
                platformFrame = platformFrame,
                captureTime = captureTime,
                leftWorldToClip = leftWorldToClip,
                rightWorldToClip = rightWorldToClip,
                leftProjectionInverse = leftProjectionInverse,
                rightProjectionInverse = rightProjectionInverse,
                leftViewInverse = leftViewInverse,
                rightViewInverse = rightViewInverse,
                leftNdc = left,
                rightNdc = right
            };
            if (_queue.TryAdd(frame))
                return true;
            ReturnArrays(frame);
            Interlocked.Decrement(ref _acceptedFrames);
            Interlocked.Increment(ref _droppedFrames);
            return false;
        }

        private void WriteLoop()
        {
            try
            {
                foreach (Frame frame in _queue.GetConsumingEnumerable())
                {
                    try
                    {
                        _writer.Write(FrameMarker);
                        _writer.Write(frame.platformFrame);
                        _writer.Write(frame.captureTime);
                        WriteMatrix(_writer, frame.leftWorldToClip);
                        WriteMatrix(_writer, frame.rightWorldToClip);
                        WriteMatrix(_writer, frame.leftProjectionInverse);
                        WriteMatrix(_writer, frame.rightProjectionInverse);
                        WriteMatrix(_writer, frame.leftViewInverse);
                        WriteMatrix(_writer, frame.rightViewInverse);
                        for (int i = 0; i < _sampleCount; i++)
                            _writer.Write(frame.leftNdc[i]);
                        for (int i = 0; i < _sampleCount; i++)
                            _writer.Write(frame.rightNdc[i]);
                        Interlocked.Increment(ref _writtenFrames);
                        if ((_writtenFrames & 7) == 0)
                            _writer.Flush();
                    }
                    finally
                    {
                        ReturnArrays(frame);
                    }
                }
                _writer.Write(EndMarker);
                _writer.Flush();
            }
            catch (Exception exception)
            {
                _failure = exception;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _queue.CompleteAdding();
            _thread.Join();
            while (_queue.TryTake(out Frame frame))
                ReturnArrays(frame);
            _queue.Dispose();
            _writer.Dispose();
        }

        private static void ReturnArrays(Frame frame)
        {
            if (frame.leftNdc != null)
                ArrayPool<float>.Shared.Return(frame.leftNdc);
            if (frame.rightNdc != null)
                ArrayPool<float>.Shared.Return(frame.rightNdc);
            frame.leftNdc = null;
            frame.rightNdc = null;
        }
    }

    public sealed class Reader : IDisposable
    {
        private readonly BinaryReader _reader;
        private readonly Frame _frame;
        private readonly int _sampleCount;
        private bool _finished;

        public int Width { get; }
        public int Height { get; }

        public Reader(string path)
        {
            _reader = new BinaryReader(new FileStream(
                path, FileMode.Open, FileAccess.Read, FileShare.Read,
                1 << 20, FileOptions.SequentialScan));
            if (_reader.ReadUInt32() != Magic)
                throw new InvalidDataException("Not a Room Setup Replica V2 tape.");
            int version = _reader.ReadInt32();
            if (version != Version)
                throw new InvalidDataException("Unsupported tape version: " + version);
            Width = _reader.ReadInt32();
            Height = _reader.ReadInt32();
            _sampleCount = checked(Width * Height);
            _frame = new Frame
            {
                leftNdc = new float[_sampleCount],
                rightNdc = new float[_sampleCount]
            };
        }

        public bool TryReadNext(out Frame frame)
        {
            frame = null;
            if (_finished)
                return false;
            uint marker;
            try
            {
                marker = _reader.ReadUInt32();
            }
            catch (EndOfStreamException)
            {
                _finished = true;
                return false;
            }
            if (marker == EndMarker)
            {
                _finished = true;
                return false;
            }
            if (marker != FrameMarker)
                throw new InvalidDataException("Corrupt V2 frame marker.");
            _frame.platformFrame = _reader.ReadInt32();
            _frame.captureTime = _reader.ReadDouble();
            _frame.leftWorldToClip = ReadMatrix(_reader);
            _frame.rightWorldToClip = ReadMatrix(_reader);
            _frame.leftProjectionInverse = ReadMatrix(_reader);
            _frame.rightProjectionInverse = ReadMatrix(_reader);
            _frame.leftViewInverse = ReadMatrix(_reader);
            _frame.rightViewInverse = ReadMatrix(_reader);
            for (int i = 0; i < _sampleCount; i++)
                _frame.leftNdc[i] = _reader.ReadSingle();
            for (int i = 0; i < _sampleCount; i++)
                _frame.rightNdc[i] = _reader.ReadSingle();
            frame = _frame;
            return true;
        }

        public void Dispose()
        {
            _reader.Dispose();
        }
    }

    private static void WriteMatrix(BinaryWriter writer, Matrix4x4 matrix)
    {
        for (int i = 0; i < 16; i++)
            writer.Write(matrix[i]);
    }

    private static Matrix4x4 ReadMatrix(BinaryReader reader)
    {
        Matrix4x4 matrix = default;
        for (int i = 0; i < 16; i++)
            matrix[i] = reader.ReadSingle();
        return matrix;
    }
}
