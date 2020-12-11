import allantoolkit as at


def test_plot():
    ds = at.dataset.Dataset(data=at.noise.white(1000), rate=1.234)
    ds.compute("adev")
    p = at.plot.Plot(no_display=True)
    p.plot(ds)
    p.show()