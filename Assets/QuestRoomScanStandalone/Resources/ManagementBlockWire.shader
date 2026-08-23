Shader "QRS/ManagementBlockWire"
{
    SubShader
    {
        Tags { "Queue"="Overlay+20" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            float _RSWireThickness;

            struct appdata
            {
                float4 vertex : POSITION;
                float4 otherAndSide : TEXCOORD0;
                fixed4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                fixed4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                float4 clipPos = UnityObjectToClipPos(v.vertex);
                float4 otherClip = UnityObjectToClipPos(float4(v.otherAndSide.xyz, 1.0));
                float thisW = max(abs(clipPos.w), 1e-5);
                float otherW = max(abs(otherClip.w), 1e-5);
                float2 thisNdc = clipPos.xy / thisW;
                float2 otherNdc = otherClip.xy / otherW;
                float2 directionPixels = (otherNdc - thisNdc) * _ScreenParams.xy;
                float directionLength = max(length(directionPixels), 1e-4);
                float2 perpendicular = float2(-directionPixels.y, directionPixels.x) /
                                       directionLength;

                // The production barycentric wire is about 0.75 px inward on
                // either side at thickness=1, i.e. roughly a 1.5 px full line.
                float halfWidthPixels = 0.75 * max(_RSWireThickness, 0.2);
                float2 ndcOffset = perpendicular * halfWidthPixels *
                                   (2.0 / _ScreenParams.xy) * v.otherAndSide.w;
                clipPos.xy += ndcOffset * clipPos.w;
                o.pos = clipPos;
                o.color = v.color;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                return i.color;
            }
            ENDCG
        }
    }
}
