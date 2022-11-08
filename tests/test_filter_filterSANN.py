import freud


def test_call_compute():
    sys = freud.data.make_random_system(10, 10)

    qargs = {"r_max": 2, "mode": "ball"}

    sann = freud.locality.FilterSANN()
    sann.compute(sys, neighbors=qargs)

    sann.filtered_nlist
    sann.unfiltered_nlist
