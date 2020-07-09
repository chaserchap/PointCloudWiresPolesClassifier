file1 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000383.las"
file2 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000384.las"
file3 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000385.las"
file4 = "/media/sf_RECON/CA_SanDiego_2005/CA_SanDiego_las/CA_SanDiego_2005_000386.las"

files = [file1, file2, file3, file4]

test_pipe = {
        "pipeline": [{"type": "filters.cluster"},
                     {"type": "filters.covariancefeatures",
                      "knn": 8,
                      "threads": 2},
                     {"type": "filters.pmf"},
                     {"type": "filters.hag"},
                     {"type": "filters.eigenvalues",
                      "knn": 16},
                     {"type": "filters.normal",
                      "knn": 16}
                     ]
    }