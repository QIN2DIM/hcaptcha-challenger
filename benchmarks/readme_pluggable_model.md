```python
# wf.py

from hcaptcha_challenger import ModelHub
# from memory_profiler import profile


modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()


@profile
def plug_model(model_name: str):
    modelhub.match_net(model_name)


@profile
def unplug_model(model_name: str):
    if model_name in modelhub._name2net:
        del modelhub._name2net[model_name]


@profile
def test_memory():
    for mn in modelhub.ashes_of_war:
        plug_model(mn)
        print("something works...")
        unplug_model(mn)

    print("that's feasible")
    print("done")


if __name__ == '__main__':
    test_memory()

```

```bash
mprof run wf.py
mprof plot -f --output pluggable_model_plot_flame_1.png
mprof plot -s --output pluggable_model_plot_1.png
```
