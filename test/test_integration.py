import iso_1D as iso
import pytest
import matplotlib.pyplot as plt
import numpy as np

TEST_BASENAME = "test/test_data/test"
TEST_BASENAME_RAYLEIGH = "test/test_data/test_rayleigh"
TEST_BASENAME_FANO = "test/test_data/test_fano"

# Initial Conditions
P0 = 100e3  # Pa
T0 = 300  # K
C_P = 1006
GAMMA = 1.4
MASSE_MOLAIRE = 28.97
ANALYTICAL_ANS = (2.84232747, 0.03454652, 0.09036560, 0.38229732)
RAYLEIGH_ANS = (1.1, 0.31309622314622315, 0.31042060261818183, 1.0087420599478942)
FANO_ANS = (1.1, 0.2797332660539672, 0.8052173429913285)


def setup_analytique():
    geo = iso.Geometry(TEST_BASENAME)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.IsoEcoulement(geo, init_cond, gas)


def iso_analytique():
    flow = setup_analytique()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def setup_rayleigh():
    geo = iso.Geometry(TEST_BASENAME_RAYLEIGH)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.General_1D_Flow(geo, init_cond, gas, basename=TEST_BASENAME_RAYLEIGH, n=2000)


def rayleigh():
    flow = setup_rayleigh()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def setup_fano():
    geo = iso.Geometry(TEST_BASENAME_FANO, friction_on=True)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.General_1D_Flow(geo, init_cond, gas, heat_on=False, n=5000)


def fano():
    flow = setup_fano()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.temperature(0.12)/T0)


def setup_iso_numerique():
    geo = iso.Geometry(TEST_BASENAME)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.General_1D_Flow(geo, init_cond, gas, heat_on=False)


def iso_numerique():
    flow = setup_iso_numerique()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def test_iso_analytique():
    assert iso_analytique() == pytest.approx(ANALYTICAL_ANS)


def test_iso_numerique():
    assert iso_numerique() == pytest.approx(ANALYTICAL_ANS, 1e-2)


def test_fano():
    assert fano() == pytest.approx(FANO_ANS, 1e-2)


def test_rayleigh():
    assert rayleigh() == pytest.approx(RAYLEIGH_ANS, 1e-2)


def main():
    flow = setup_fano()
    x_array = np.linspace(flow.geo.x_points[0], flow.geo.x_points[-1], 1000)
    print(flow.geo.D_hydraulique(0.05))

    plt.plot(x_array, flow.mach(x_array))
    plt.show()



if __name__ == "__main__":
    main()