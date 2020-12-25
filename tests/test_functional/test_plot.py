import allantoolkit as at


def test_plot():

    ds = at.dataset.Dataset(data=at.noise.white(1000), rate=1.234,
                            data_type='freq')

    ds.calc("adev", data_type='freq')

    p = at.plot.Plot(no_display=True)
    p.plot(ds, grid=True)