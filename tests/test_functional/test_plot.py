import allantoolkit as at


def test_plot():

    ds = at.api.Dataset(data=at.noise.white(1000), rate=1.234,
                        data_type='freq')

    ds.calc("adev")

    p = at.plot.Plot(no_display=True)
    p.plot(ds, grid=True)