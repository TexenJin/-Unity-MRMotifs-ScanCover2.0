using System;
using System.Reflection;
using UnityEditor;

namespace MRMotifs.InstantContentPlacement.Editor
{
    [InitializeOnLoad]
    internal static class DepthEffectsPlayModeAnnotationSuppressor
    {
        private static readonly Type AnnotationUtilityType =
            Type.GetType("UnityEditor.AnnotationUtility, UnityEditor");

        private static readonly MethodInfo GetAnnotationsMethod =
            AnnotationUtilityType?.GetMethod("GetAnnotations", BindingFlags.Static | BindingFlags.NonPublic);

        private static readonly MethodInfo SetIconEnabledMethod =
            AnnotationUtilityType?.GetMethod("SetIconEnabled", BindingFlags.Static | BindingFlags.NonPublic);

        static DepthEffectsPlayModeAnnotationSuppressor()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.EnteredPlayMode)
                DisableAllAnnotationIcons();
        }

        [MenuItem("Tools/DepthEffects/Disable Annotation Icons In Play")]
        private static void DisableAllAnnotationIcons()
        {
            if (AnnotationUtilityType == null || GetAnnotationsMethod == null || SetIconEnabledMethod == null)
                return;

            object annotations = GetAnnotationsMethod.Invoke(null, null);
            Array annotationArray = annotations as Array;
            if (annotationArray == null)
                return;

            foreach (object annotation in annotationArray)
            {
                if (annotation == null)
                    continue;

                Type annotationType = annotation.GetType();
                int classId = (int)(annotationType.GetField("classID")?.GetValue(annotation) ?? 0);
                string scriptClass = (string)(annotationType.GetField("scriptClass")?.GetValue(annotation) ?? string.Empty);
                SetIconEnabledMethod.Invoke(null, new object[] { classId, scriptClass, 0 });
            }
        }
    }
}
