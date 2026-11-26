# Real Test Data Cache

This ignored directory can cache optional MRI integration-test data. HyperMRI
does not commit or redistribute scanner data here.

The approved test entry is Tomáš Pšorn's public **BrukerAPI test data set
(Bruker ParaVision v6.0.1)**, version 1.1.0, licensed under CC BY 4.0:

- DOI: <https://doi.org/10.5281/zenodo.4048253>
- manifest key: `brukerapi_pv601_lego_phantom`
- compressed size: 783,667,029 bytes
- extracted directory: `20200612_094625_lego_phantom_3_1_2/`

Download it explicitly with:

```bash
python scripts/download_test_data.py --dataset brukerapi_pv601_lego_phantom
```

Set `HYPERMRI_TEST_DATA` to use a cache outside the repository. The public
phantom archive still contains operator names, workstation paths, timestamps,
UIDs, and hardware identifiers. Do not commit, quote, or print those metadata.
