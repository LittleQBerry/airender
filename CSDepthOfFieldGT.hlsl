//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
RWTexture2D<float4> g_rwDst;

Texture2D<float3> g_txSrc;
Texture2D<float> g_txCoc;

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	// Bilateral filter
	float4 dof = 0.0;
	for (int y = -BOKEH_RADIUS; y < BOKEH_RADIUS; ++y)
	{
		for (int x = -BOKEH_RADIUS; x < BOKEH_RADIUS; ++x)
		{
			const int2 offset = int2(x, y);
			const uint2 idx = (int2)DTid + offset;

			float coc = g_txCoc[idx];
			const float3 src = g_txSrc[idx];

			coc = abs(coc);
			const float radius = length(offset);
			float w = CoCWeight(coc, radius);

			w /= max(coc * coc, 1.0);
			dof.xyz += src * w;
			dof.w += w;
		}
	}

	g_rwDst[DTid] = float4(dof.w > 0.0 ? dof.xyz / dof.w : dof.xyz, 1.0);
}
