import unittest
import pandas as pd
from furcast_data import (build_austin,connected_groups,feature_guard,temporal_split,local_times,age_days,prepare_petfinder,PF_REQUIRED)

def intake(animal,time):
    return dict(animal_id=animal,datetime=time,animal_type='Dog',intake_type='Stray',intake_condition='Normal',
                sex_upon_intake='Intact Male',breed='Mix',color='Black',age_upon_intake='2 years')
def outcome(animal,time,kind='Adoption'):
    return dict(animal_id=animal,datetime=time,outcome_type=kind)
def build(ii,oo):
    return build_austin(pd.DataFrame(ii),pd.DataFrame(oo),'legacy','2025-05-05T00:00:00-05:00')

class Guards(unittest.TestCase):
    def test_repeat_admissions_do_not_cartesian_join(self):
        d,_,_=build([intake('A','2024-01-01'),intake('A','2024-03-01')],
                    [outcome('A','2024-01-03'),outcome('A','2024-03-04')])
        self.assertEqual(d.duration_days.tolist(),[2,3])
    def test_transfer_not_adoption(self):
        d,_,_=build([intake('A','2024-01-01')],[outcome('A','2024-02-03','Transfer')])
        self.assertEqual(d.event_cause.iloc[0],'other_exit');self.assertEqual(d.long_stay_30d.iloc[0],1)
    def test_multiple_outcomes_quarantined(self):
        d,_,_=build([intake('A','2024-01-01')],[outcome('A','2024-01-03'),outcome('A','2024-01-04')])
        self.assertTrue(d.long_stay_30d.isna().all())
    def test_unmatched_not_assumed_censored(self):
        d,_,_=build([intake('A','2024-01-01')],[outcome('B','2024-01-04')])
        self.assertEqual(d.status.iloc[0],'unmatched_unverified');self.assertTrue(d.long_stay_30d.isna().all())
    def test_future_outcome_not_used(self):
        d,_,_=build([intake('A','2025-05-01')],[outcome('A','2025-06-01')])
        self.assertFalse(d.event_observed.any())
    def test_ambiguous_clock_quarantines_timeline(self):
        d,_,_=build([intake('A','2024-01-01')],[outcome('A','2024-01-03T00:00:00-05:00')])
        self.assertEqual(d.status.iloc[0],'quarantine_timeline')
    def test_exact_duplicate_event_dedup(self):
        o=outcome('A','2024-01-03')
        d,_,_=build([intake('A','2024-01-01')],[o,o])
        self.assertTrue(d.event_observed.all())
    def test_transitive_rescuer_duplicate_groups(self):
        d=pd.DataFrame({'id':['a','b','c','d'],'rescuer':['r1','r1','r2','r3']})
        e=pd.DataFrame({'id_a':['b'],'id_b':['c']})
        g=connected_groups(d,'id','rescuer',e)
        self.assertEqual(g.nunique(),2);self.assertEqual(g.iloc[0],g.iloc[2])
    def test_leaky_feature_rejected(self):
        for col in ['outcome_time','days_in_shelter','age_upon_outcome','outcome_type','animal_id']:
            with self.assertRaises(ValueError):feature_guard([col],'austin')
    def test_age_and_dst(self):
        self.assertEqual(age_days(pd.Series(['2 weeks'])).iloc[0],14)
        self.assertTrue(local_times(pd.Series(['2024-11-03T01:30:00']))[1].iloc[0])
    def test_temporal_labels_and_repeated_animals(self):
        ii=[intake('A','2022-01-01'),intake('B','2023-01-01'),intake('C','2024-01-01'),
            intake('D','2022-01-01'),intake('D','2023-01-01'),intake('E','2022-12-30')]
        oo=[outcome('A','2022-01-03'),outcome('B','2023-01-03'),outcome('C','2024-01-03'),
            outcome('D','2022-01-03'),outcome('D','2023-01-03'),outcome('E','2023-01-04')]
        d,_,_=build(ii,oo);s=temporal_split(d,'2023-01-01T00:00:00-06:00','2024-01-01T00:00:00-06:00')
        self.assertTrue(s.loc[s.animal_id.isin(['D','E']),'split'].eq('excluded').all())
        self.assertEqual(s.loc[s.animal_id.eq('A'),'split'].iloc[0],'train')
    def test_current_negative_and_exact_join(self):
        ii=pd.DataFrame([dict(id='i1',animal_id='A',source_date='2025-06-01T10:00:00',type='Puppy',source_name='Stray',
            intake_health_condition='Normal',sex='Male',primary_breed='Mix',primary_color='Black',date_of_birth='2025-01-01')])
        oo=pd.DataFrame([dict(id='o1',animal_id='A',intake_date='2025-06-01T10:00:00',outcome_date='2025-06-01T09:00:00',outcome_status='Adopted',days_in_shelter=0)])
        d,_,_=build_austin(ii,oo,'current','2026-10-11T19:00:00-05:00')
        self.assertEqual(d.status.iloc[0],'quarantine_time_order')

if __name__=='__main__':unittest.main(verbosity=2)
