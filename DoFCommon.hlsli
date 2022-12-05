//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#define BOKEH_RADIUS 32

float CoCWeight(float coc, float radius)
{
	//return radius <= coc ? 1.0 : 0.0;
	return saturate((coc - radius + 2.0) / 2.0);
}

float DepthWeight(float depthC, float depth, float sigma)
{
	return exp(-abs(depthC - depth) * depthC * sigma);
}

float Gaussian(float x, float m, float sigma)
{
	const float r = x - m;
	const float a = r * r / (sigma * sigma);

	return exp(-0.5 * a);
}
