/*
* BM3D denoising filter - VapourSynth plugin
* Copyright (C) 2015  mawen1250
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#if  defined(__SSE2__)
inline static __m128 _mm_abs_ps(__m128 x)
{
    static __m128 const Mask = _mm_castsi128_ps(_mm_set1_epi32(~0x80000000));

    __m128 abs = _mm_and_ps(Mask, x);

    return abs;
}
#elif defined(__AVX2__)
inline static __m256 _mm256_abs_ps(__m256 x)
{
    static __m256 const Mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000));

    __m256 abs = _mm256_and_ps(Mask, x);

    return abs;
}
#endif
union FI{
float *f;
uint32_t *i;
};

inline float ffabs(float a)
{
    FI fi;
    fi.f = &a;
    *fi.i &= (~0x80000000);

    return *fi.f;
}

#include "BM3D_Basic.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D_Basic_Data


int BM3D_Basic_Data::arguments_process(const VSMap *in, VSMap *out)
{
    if (_Mybase::arguments_process(in, out))
    {
        return 1;
    }

    int error;

    // hard_thr - float
    para.lambda = vsapi->propGetFloat(in, "hard_thr", 0, &error);

    if (error)
    {
        para.lambda = para_default.lambda;
    }
    else if (para.lambda <= 0)
    {
        setError(out, "Invalid \"hard_thr\" assigned, must be a positive floating point number");
        return 1;
    }

    // Initialize filter data for hard-threshold filtering
    init_filter_data();

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D_Basic_Process


void BM3D_Basic_Process::CollaborativeFilter(int plane,
    FLType *ResNum, FLType *ResDen,
    const FLType *src, const FLType *ref,
    const PosPairCode &code) const
{
    PCType GroupSize = static_cast<PCType>(code.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (d.para.GroupSize > 0 && GroupSize > d.para.GroupSize)
    {
        GroupSize = d.para.GroupSize;
    }

    // Construct source group guided by matched pos code
    block_group srcGroup(src, src_stride[plane], code, GroupSize, d.para.BlockSize, d.para.BlockSize);

    // Initialize retianed coefficients of hard threshold filtering
    int retainedCoefs = 0;

    // Apply forward 3D transform to the source group
    d.f[plane].fp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Apply hard-thresholding to the source group
    auto srcp = srcGroup.data();
    auto thrp = d.f[plane].thrTable[GroupSize - 1].get();
    const auto upper = srcp + srcGroup.size();

#if defined(__SSE2__)
    static const ptrdiff_t simd_step = 4;
    const ptrdiff_t simd_residue = srcGroup.size() % simd_step;
    const ptrdiff_t simd_width = srcGroup.size() - simd_residue;

    static const __m128 zero_ps = _mm_setzero_ps();
    __m128i cmp_sum = _mm_setzero_si128();

    for (const auto upper1 = srcp + simd_width; srcp < upper1; srcp += simd_step, thrp += simd_step)
    {
        const __m128 s1 = _mm_load_ps(srcp);
        const __m128 t1p = _mm_load_ps(thrp);
		
		const __m128 abs = _mm_abs_ps(srcp);
		const __m128 cmp = _mm_cmpgt_ps(s1, t1p);
		
        const __m128 d1 = _mm_and_ps(cmp, s1);
        _mm_store_ps(srcp, d1);
        cmp_sum = _mm_sub_epi32(cmp_sum, _mm_castps_si128(cmp));
    }

    alignas(16) int32_t cmp_sum_i32[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(cmp_sum_i32), cmp_sum);
    retainedCoefs += cmp_sum_i32[0] + cmp_sum_i32[1] + cmp_sum_i32[2] + cmp_sum_i32[3];
	
#elif defined(__AVX2__)
    static const ptrdiff_t simd_step = 8;
    const ptrdiff_t simd_residue = srcGroup.size() % simd_step;
    const ptrdiff_t simd_width = srcGroup.size() - simd_residue;
	
    __m128i cmp_sum   = _mm_setzero_si128();
	
	__m128i cmp_0, cmp_1;
	
    for (const auto upper1 = srcp + simd_width; srcp < upper1; srcp += simd_step, thrp += simd_step)
    {
        const __m256 s1 = _mm256_load_ps(srcp);
        const __m256 t1 = _mm256_load_ps(thrp);
		
        const __m256 abs = _mm256_abs_ps(s1);
        const __m256 cmp = _mm256_cmp_ps(abs, t1, _CMP_GT_OQ);

        const __m256 d1 = _mm256_and_ps(cmp, s1);
        _mm256_storeu_ps(srcp, d1);
		
        cmp_sum = _mm256_sub_epi32(cmp_sum, _mm256_castps_si256(cmp));
    }

    alignas(32) int32_t cmp_sum_i32[8];
    _mm_store_si128(reinterpret_cast<__m128i *>(cmp_sum_i32), cmp_sum);
    retainedCoefs += cmp_sum_i32[0] + cmp_sum_i32[1] + cmp_sum_i32[2] + cmp_sum_i32[3]\
					 cmp_sum_i32[4] + cmp_sum_i32[5] + cmp_sum_i32[6] + cmp_sum_i32[7];
#endif


    for (; srcp < upper; ++srcp, ++thrp)
    {
        if (ffabs(*srcp) > *thrp)
        {
            ++retainedCoefs;
        }
        else
        {
            *srcp = 0;
        }
    }

    // Apply backward 3D transform to the filtered group
    d.f[plane].bp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Calculate weight for the filtered group
    // Also include the normalization factor to compensate for the amplification introduced in 3D transform
    FLType denWeight = retainedCoefs < 1 ? 1 : FLType(1) / static_cast<FLType>(retainedCoefs);
    FLType numWeight = static_cast<FLType>(denWeight / d.f[plane].finalAMP[GroupSize - 1]);

    // Store the weighted filtered group to the numerator part of the basic estimation
    // Store the weight to the denominator part of the basic estimation
    srcGroup.AddTo(ResNum, dst_stride[plane], numWeight);
    srcGroup.CountTo(ResDen, dst_stride[plane], denWeight);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
