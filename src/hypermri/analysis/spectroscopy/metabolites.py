"""Retained metabolite chemical-shift references in ppm."""

__all__ = ["METABOLITE_CS_PPM", "get_metab_cs_ppm"]


METABOLITE_CS_PPM = {
    "pyruvate": 171.076,  # Lactate - 920Hz
    "lactate": 183.35,  # determined from repeated experiments
    "alanine": 176.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "pyruvatehydrate": 179.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "fumarate": 175.4,
    "malate1": 181.7,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    "malate4": 180.5,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    # https://pubmed.ncbi.nlm.nih.gov/30639333/ (fits human in vivo data @
    # 3T)
    "bicarbonate": 161.0,
    "urea": 163.5,  # trust me bro
    "co2": 124.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "aspartate1": 176.92,  # doi: 10.3389/fphys.2021.792769
    "aspartate4": 180.20,  # doi: 10.3389/fphys.2021.792769
}


def get_metab_cs_ppm(metab: str = None, help: bool = False) -> float | None:
    """Look up a retained metabolite chemical-shift reference.

    Parameters
    ----------
    metab : str or None, optional
        Case-insensitive metabolite identifier in METABOLITE_CS_PPM. None is
        supported only with help=True to list the references.
    help : bool, optional
        Print available identifiers when no metabolite or an unknown one is given.

    Returns
    -------
    float or None
        Reference in ppm, or None for an unknown identifier or a printed listing.
        Values are legacy references, without sample-specific pH/temperature
        calibration. The mapping is not mutated; lookup is deterministic.

    Raises
    ------
    AttributeError
        If metab is None and help=False, under the retained legacy contract.

    Examples
    --------
    >>> get_metab_cs_ppm("pyruvate")
    171.076
    """
    if metab == "" and help is True:
        pass
    elif metab != "":
        pass
    # valid nuclei:

    if metab is None and help is True:
        for m in METABOLITE_CS_PPM.keys():
            print(f"{m}: {METABOLITE_CS_PPM[m]} ppm")
        return None

    else:
        # Convert metab to title case (first letter capitalized)
        metab = metab.lower()

        # Checking for valid metabolite
        if metab in METABOLITE_CS_PPM:
            return METABOLITE_CS_PPM[metab]
        elif metab not in METABOLITE_CS_PPM and help is True:
            print(f"Invalid metabolite {metab}. Available metabolites are:")
            for m in METABOLITE_CS_PPM.keys():
                print(m)
            return None
        else:
            return None
