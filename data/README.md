# data README
from: https://osf.io/ry3en/files/?view_only=39da1fe1f24c4c8191007d60f3364af0

18 January, 2019
Eleanor Chodroff, Alessandra Golden, and Colin Wilson


###########################
Please cite the following for use of this data: 
Chodroff, E., Golden, A., and Wilson, C. (2019). Cross-linguistic covariation of stop VOT: Evidence for a universal constraint on phonetic realization: covariation of stop voice onset time across languages. The Journal of the Acoustical Society of America, 145(1). EL1–EL7.

@article{ChodroffEtAl2019,
author = {Chodroff, Eleanor and Golden, Alessandra and Wilson, Colin},
doi = {10.1121/1.5088035},
journal = {The Journal of the Acoustical Society of America},
number = {1},
pages = {EL1--EL7},
title = {Covariation of stop voice onset time across languages : {E}vidence for a universal constraint on phonetic realization},
volume = {145},
year = {2019}
}
########################### 

This directory accompanies the manuscript "Cross-linguistic covariation of stop VOT: Evidence for a universal constraint on phonetic realization: covariation of stop voice onset time across languages" in JASA Express Letters.

The directory contains the following items:

1) ChodroffGoldenWilson2019_vot.csv: The VOT values collected from the literature. The values are typically averages over tokens and frequently groups of speakers, though the primary source should be consulted for verification. The columns within this dataset are as follows:
	family: language family obtained from WALS
	language: obtained from source
	dialect: obtained from source, if specified
	source: location of collected VOT value, please refer to *_references.pdf for the full citations
	primarySource: original location of collected VOT value (NA indicates that the source is indeed the primary source)
	poa.narrow: narrow place of articulation, provided by the source
	poa.broad: broad place of articulation (labial, coronal, dorsal), specified by authors, please refer to primary text for description
	voice: voice specification from the source
	vot.category: broad VOT category (long.lag, short.lag, lead), specified by authors, please refer to primary text for description
	vot: voice onset time (VOT) value
	voiceContrast: yes/no indicating whether the language has a voice contrast, obtained from WALS or in cases of omission, from the source of the VOT value
	notes: additional notes regarding the materials or speakers in the cited study
	

2) ChodroffGoldenWilson2019_vot_avg.csv: The VOT means averaged within each language-dialect pairing for each place of articulation and VOT category (long-lag, short-lag, lead). The columns within this dataset are as follows:
	family: language family obtained from WALS
	language: obtained from source
	dialect: obtained from source, if specified
	poa.broad: broad place of articulation (labial, coronal, dorsal), specified by authors based on poa.narrow (see ChodroffGoldenWilson2019_vot.csv)
	vot.category: broad VOT category (long.lag, short.lag, lead), specified by authors, please refer to primary text for description
	vot.mu: voice onset time (VOT) mean within each language-dialect pairing per broad place of articulation and vot.category


3) ChodroffGoldenWilson2019_references.pdf: References for the collected VOT values.
