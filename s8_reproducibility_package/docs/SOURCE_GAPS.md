# Source gaps and provenance notes

The files available to this consolidation step include the S1 result CSV and
S2 result CSVs, but the original S1 and S2 Python source scripts were not
present in the supplied artifact set used to prepare this package. The S8
script searches the user's project directory for `s1*.py` and `s2*.py` and
will include them if present. If absent, they remain explicitly marked as
missing source, while their supplied result artifacts are retained.

The package does not claim that result CSVs can replace the original scripts.
Likewise, a successful S4-S7 validation report is evidence for those stored
outputs, not proof that every historical stage can be regenerated from source.
