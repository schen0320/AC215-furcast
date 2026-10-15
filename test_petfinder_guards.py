"""Meaningful regression checks for radius search and post-outcome exclusion."""
import random
import unittest
import pandas as pd
from find_petfinder_duplicates import find_pairs
from prepare_petfinder_real import FEATURES, validate_table


class PetFinderGuards(unittest.TestCase):
    def test_radius_search_matches_bruteforce_including_band_boundaries(self):
        rng=random.Random(215);values=[rng.getrandbits(64) for _ in range(50)]
        for base in list(values[:10]):
            for distance in range(9):
                value=base
                for bit in rng.sample(range(64),distance):value^=1<<bit
                values.append(value)
        frame=pd.DataFrame({'key':['x:'+str(i) for i in range(len(values))],
            'phash':[f'{v:016x}' for v in values],'pixel_sha256':[str(i) for i in range(len(values))]})
        # Exact pixels must still link even with a corrupted/inconsistent pHash.
        frame.loc[len(frame)-1,'pixel_sha256']=frame.loc[0,'pixel_sha256']
        actual={frozenset([r.id_a,r.id_b]) for r in find_pairs(frame).itertuples()}
        expected={frozenset([frame.key[i],frame.key[j]]) for i in range(len(values)) for j in range(i)
                  if (values[i]^values[j]).bit_count()<=6 or frame.pixel_sha256[i]==frame.pixel_sha256[j]}
        self.assertEqual(actual,expected)

    def test_feature_allowlist_excludes_outcomes_and_unverified_mutable_fields(self):
        forbidden={'AdoptionSpeed','slow_listing_risk_label','Description','PhotoAmt','VideoAmt',
                   'Health','Sterilized','Vaccinated','Dewormed','Fee','PetID','RescuerID','split'}
        self.assertFalse(set(FEATURES)&forbidden)

    def test_missing_real_schema_is_rejected(self):
        with self.assertRaises(ValueError):validate_table(pd.DataFrame({'PetID':['x']}),True)


if __name__=='__main__':unittest.main()
