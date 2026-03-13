# Dear ImGui Integration in Xylem

This document explains how the immediate-mode GUI (Dear ImGui) is wired into
the Xylem renderer through Donut's `ImGui_Renderer` abstraction.

---

## The execution model

Dear ImGui is an *immediate-mode* GUI library.  Unlike retained-mode frameworks
(WinForms, Qt, etc.) where you create widget objects and then receive events
from them, ImGui works the opposite way:

- **Every frame**, you call functions like `ImGui::SliderFloat(...)` or
  `ImGui::Button(...)`.
- Each call *both* draws the widget and *returns its current value or whether
  it was activated*.
- There is no persistent widget state beyond what you store yourself.

The loop looks like this each frame:

```
1. ImGui::NewFrame()           ← Donut calls this for you
2.   buildUI()                 ← Your code; call all the ImGui widget functions
3. ImGui::Render()             ← Donut calls this for you
4. GPU draws the ImGui mesh    ← Donut submits a command list for you
```

---

## Donut's abstraction: `app::ImGui_Renderer`

Donut wraps ImGui in the class `donut::app::ImGui_Renderer` found in
`donut/include/donut/app/imgui_renderer.h`.  It is itself an `IRenderPass` and
must be registered with `DeviceManager` *after* your scene render pass so it
composites on top.

```cpp
deviceManager->AddRenderPassToBack(&renderPass);  // scene drawn first
deviceManager->AddRenderPassToBack(&uiPass);      // UI drawn on top
```

To use it, subclass `ImGui_Renderer` and override `buildUI()`:

```cpp
class XylemUIRenderer : public app::ImGui_Renderer {
protected:
    void buildUI() override {
        ImGui::Begin("My Window");
        ImGui::Text("Hello, Xylem!");
        ImGui::End();
    }
};
```

Call `ImGui_Renderer::Init(shaderFactory)` after construction to compile the
ImGui shaders.  You need a `ShaderFactory` that mounts the Donut framework
shader path at `/shaders/donut` because ImGui's vertex/pixel shaders live
there.

---

## Passing data between the render pass and the UI

### The `UIData` struct

A plain struct, allocated on the stack in `main()` and passed by reference to
both the render pass and the UI renderer.  This is the *communication channel*:

```
main()
  UIData uiData;
  TraditionalRenderPass renderPass(dm, uiData);   ← writes stats into uiData
  XylemUIRenderer       uiPass(dm, &renderPass, uiData); ← reads stats, writes flags
```

- The *render pass* writes performance numbers (`gpuFrameTimeMs`,
  `visibleInstanceCount`, …) into `UIData` every frame.
- The *UI renderer* reads those numbers and displays them.
- The *UI renderer* also calls mutator methods on the render pass directly
  (`m_pass->AddTreeAsset(...)`, `m_pass->RemoveRegion(...)`).

### Why not pass raw pointers everywhere?

`UIData` only holds *cheap scalar values*.  Large data (assets, regions,
L-Systems) lives inside `TraditionalRenderPass` and is accessed through
its public getter/mutator API.  This keeps ownership clear: the render
pass *owns* its scene data.

---

## Widgets and when they trigger GPU work

Most ImGui widgets are harmless — they just display text or let you drag a
slider.  Some widgets in Xylem trigger expensive GPU operations:

| Widget action             | What it does                                       |
|---------------------------|----------------------------------------------------|
| "Add Asset"               | Opens a command list, uploads vertex/index buffers |
| "Update Asset"            | Releases old buffers, re-uploads new geometry      |
| "Remove Asset"            | Releases NVRHI handles (GPU memory freed lazily)   |
| "Add Region"              | Rebuilds the instance buffer on the GPU            |
| "Update Region"           | Rebuilds the instance buffer on the GPU            |
| "Load Scene"              | Full rebuild of all GPU resources                  |

Because of this, **never call mutators from inside the ImGui draw loop
on every frame**.  Only call them in response to an `ImGui::Button` or
similar one-shot event.

---

## The edit-state structs

The UI renderer keeps small POD structs like `AssetEditState` and
`RegionEditState` as member variables.  These exist purely to hold the
"draft" values while the user is typing/dragging before they commit with
an "Add" or "Update" button.

```cpp
struct AssetEditState {
    char     name[128] = "NewAsset";
    uint32_t generation = 3;
    float    branchAngle = 25.f;
    bool     editing = false;   // true = we are editing an existing asset
    size_t   editingIndex = 0;  // which asset is being edited
};
```

When the user presses "Edit" on an existing asset, the edit-state fields are
populated from the asset's current values.  When they press "Update", those
fields are sent to `UpdateTreeAsset()`.

This pattern is idiomatic ImGui: the *UI owns the draft*; the *render pass
owns the authoritative data*.

---

## Memory management and the `new T(val)` pitfall

`ImGui::SliderScalar` takes `const void* p_min, const void* p_max` pointers.
A common mistake is:

```cpp
// BAD: leaks memory every frame
ImGui::SliderScalar("Gen", ImGuiDataType_U32, &val, new uint32_t(1), new uint32_t(10));
```

The correct pattern is to use `static constexpr` values:

```cpp
static constexpr uint32_t kMin = 1, kMax = 10;
ImGui::SliderScalar("Gen", ImGuiDataType_U32, &val, &kMin, &kMax);
```

---

## Toggling the UI

The `GLFW_KEY_ESCAPE` handler in `XylemUIRenderer::KeyboardUpdate` flips
`m_ui.ShowUI`.  The `buildUI()` function returns immediately when that flag
is false, so ImGui produces an empty draw list — no quads are rendered.

---

## Performance impact of ImGui

ImGui itself is cheap: a few thousand vertices per frame.  The only place it
becomes expensive is if widget callbacks trigger GPU resource creation.
The performance panel added to the UI (`gpuFrameTimeMs`, `cpuRenderTimeMs`,
FPS counter) lets you see the true frame cost of Xylem's pipelines independently
of ImGui overhead.

---

## Adding a new panel

1. Add any new fields needed to `UIData` (for readouts) or `XylemUIRenderer`
   (for draft-edit state).
2. Inside `buildUI()`, add a new `ImGui::CollapsingHeader(...)` block.
3. If the panel triggers render-pass mutations, add a corresponding public method
   to `TraditionalRenderPass` (or the appropriate pipeline render pass).

For the future mesh-shader or work-graph pipelines, each pipeline's render pass
will expose its own set of parameters.  A recommended pattern is to have the
UI renderer hold a `std::variant` or a simple enum-driven union of per-pipeline
edit states, switching on the currently active pipeline.
