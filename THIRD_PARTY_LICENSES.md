# Third-Party Licenses

This file lists the model artifacts LibreImage downloads and the direct
dependencies declared in `pyproject.toml`. Transitive dependencies are governed
by their own package metadata and license files.

## Downloaded Models

| Component | Source | Local path | License |
| --- | --- | --- | --- |
| FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | `models/FLUX.1-Kontext-dev` | Flux-1 Dev Non-Commercial License (`license: other`, `license_name: flux-1-dev-non-commercial-license`) |
| Stable Diffusion XL Inpainting 0.1 | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | `models/stable-diffusion-xl-1.0-inpainting-0.1` | OpenRAIL++ |
| Real-ESRGAN x2 | `ai-forever/Real-ESRGAN` | `models/Real-ESRGAN/RealESRGAN_x2.pth` | BSD 3-Clause, per upstream Real-ESRGAN repository |

Model license references:

- FLUX.1-Kontext-dev model card:
  https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- FLUX.1-Kontext-dev license:
  https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md
- Stable Diffusion XL Inpainting model card:
  https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- Real-ESRGAN model card:
  https://huggingface.co/ai-forever/Real-ESRGAN
- Real-ESRGAN upstream license:
  https://github.com/ai-forever/Real-ESRGAN/blob/main/LICENSE

## Direct Runtime Dependencies

| Dependency | Declared requirement | License |
| --- | --- | --- |
| `accelerate` | `accelerate>=0.31` | Apache-2.0 |
| `diffusers` | `diffusers>=0.36` | Apache-2.0 |
| `fastapi` | `fastapi>=0.111` | MIT |
| `huggingface-hub` | `huggingface-hub>=0.23` | Apache-2.0 |
| `numpy` | `numpy>=1.26` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `pillow` | `pillow>=10.3` | MIT-CMU |
| `python-multipart` | `python-multipart>=0.0.9` | Apache-2.0 |
| `safetensors` | `safetensors>=0.4` | Apache-2.0 |
| `torch` | `torch>=2.3` | BSD-3-Clause |
| `transformers` | `transformers>=4.41` | Apache-2.0 |
| `uvicorn[standard]` | `uvicorn[standard]>=0.30` | BSD-3-Clause |

## Direct Build Dependencies

| Dependency | Declared requirement | License |
| --- | --- | --- |
| `hatchling` | `hatchling` | MIT |

The runtime dependency license values above come from installed package metadata
for the direct dependencies. The build dependency license is from its published
PyPI metadata. For packages with compound license expressions, preserve the
package's own license files when redistributing.

## Vendored Real-ESRGAN Architecture Code

The RRDBNet architecture in `src/libreimage/realesrgan.py` is adapted from
`ai-forever/Real-ESRGAN`.

BSD 3-Clause License

Copyright (c) 2021, Sberbank AI
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
