# Third-party notices

ORBI StreamMoE is informed by and interoperates with open-source projects. Model weights are not distributed in this repository.

## Swiftlet

Swiftlet by Léon Simmons / contributors  
Repository: https://github.com/leonickson1/Swiftlet  
License: Apache License 2.0

ORBI StreamMoE uses Swiftlet as a design reference and qpack compatibility target. The initial audit is pinned to commit `909c04213c9deb369dac0679d0872512cf3ab32e`.

## nlohmann/json

JSON for Modern C++  
Repository: https://github.com/nlohmann/json  
License: MIT

Used to parse qpack manifest and layout JSON metadata.


## Vulkan-Headers

Vulkan API headers and registry by The Khronos Group Inc.  
Repository: https://github.com/KhronosGroup/Vulkan-Headers  
Compatibility pin: `v1.4.363`  
Licensing: files are Apache-2.0 and/or MIT as declared by upstream.

Used as the platform-neutral Vulkan API contract.

## volk

volk by Arseny Kapoulkine / contributors  
Repository: https://github.com/zeux/volk  
Compatibility pin: `vulkan-sdk-1.4.363`  
License: MIT

Used to dynamically load Vulkan entry points on Windows, Linux and Android without a hard link-time Vulkan-loader dependency.
