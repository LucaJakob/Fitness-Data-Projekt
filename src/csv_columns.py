from itertools import chain

class _IterableAttributes:
    def __get_attributes(self):
        return [
            getattr(self, k) for k in dir(self) 
            if not k.startswith('_') # filter out hidden props
            and not callable(getattr(self, k)) # no methods
            and type(getattr(self, k)) is not staticmethod # no statics
        ]

    def __iter__(self):
        return iter(self.__get_attributes())

    def __len__(self):
        return len(self.__get_attributes())
    
# Represents any columns labeled as unknown or unnamed
class _Invalid(_IterableAttributes):
    unnamed_0 = 'Unnamed: 0'
    unnamed_33 = 'Unnamed: 33'
    unnamed_51 = 'Unnamed: 51'
    unnamed_55 = 'Unnamed: 55'
    unnamed_53 = 'Unnamed: 53'
    empty = ''
    unknown = 'unknown.unknown'
    monitoring_info_unknown = 'monitoring_info.unknown'
    monitoring_unknown = 'monitoring.unknown'
    stress_level_unknown = 'stress_level.unknown'
    sleep_assessment_unknown = 'sleep_assessment.unknown'

class _FileId(_IterableAttributes):
    serial_number = 'file_id.serial_number'
    time_created = 'file_id.time_created'
    manufacturer = 'file_id.manufacturer'
    garmin_product = 'file_id.garmin_product'
    number = 'file_id.number'
    file_type = 'file_id.type'

class _DeviceInfo(_IterableAttributes):
    timestamp_s = 'device_info.timestamp[s]'
    serial_number = 'device_info.serial_number'
    manufacturer = 'device_info.manufacturer'
    garmin_product = 'device_info.garmin_product'
    software_version = 'device_info.software_version'
	

class _Software(_IterableAttributes):
    version = 'software.version'
    file_creator_version = 'file_creator.software_version'

class _MonitoringInfo(_IterableAttributes):
    timestamp_s = 'monitoring_info.timestamp[s]'
    local_timestamp_s = 'monitoring_info.local_timestamp[s]'
    meters_per_cycle = 'monitoring_info.cycles_to_distance[m/cycle]'
    calories_per_cycles = 'monitoring_info.cycles_to_calories[kcal/cycle]'
    resting_metabolic_rate = 'monitoring_info.resting_metabolic_rate[kcal / day]'
    activity_type = 'monitoring_info.activity_type'

class _Monitoring(_IterableAttributes):
    timestamp_s = 'monitoring.timestamp[s]'
    timestamp16_s = 'monitoring.timestamp_16[s]'
    bpm = 'monitoring.heart_rate[bpm]'
    activity_type_intensity = 'monitoring.current_activity_type_intensity'
    activity_type = 'monitoring.activity_type' 
    intensity = 'monitoring.intensity'
    steps = 'monitoring.steps[steps]'
    active_time_s = 'monitoring.active_time[s]'
    active_kcal = 'monitoring.active_calories[kcal]' 
    cycles = 'monitoring.cycles[cycles]'
    ascent_m = 'monitoring.ascent[m]'
    descent_m = 'monitoring.descent[m]'
    distance_m = 'monitoring.distance[m]'
    duration_min = 'monitoring.duration_min[min]'
    vigorous_activity_minutes = 'monitoring.vigorous_activity_minutes[minutes]'
    moderate_activity_minutes = 'monitoring.moderate_activity_minutes[minutes]'

class _MonitoringHRData(_IterableAttributes):
    timestamp_s = 'monitoring_hr_data.timestamp[s]'
    resting_bpm = 'monitoring_hr_data.resting_heart_rate[bpm]'
    current_day_resting_bpm = 'monitoring_hr_data.current_day_resting_heart_rate[bpm]'

class _OhrSettings(_IterableAttributes):
    timestamp_s = 'ohr_settings.timestamp[s]'
    enabled = 'ohr_settings.enabled' 

class _Event(_IterableAttributes):
    timestamp_s = 'event.timestamp[s]'
    data16 = 'event.data16'
    event = 'event.event'
    event_type = 'event.event_type'
    data = 'event.data'
    auto_activity_detect_start_timestamp_s = 'event.auto_activity_detect_start_timestamp[s]' 
    activity_type = 'event.activity_type'
    auto_activity_detect_duration_min = 'event.auto_activity_detect_duration[min]' 


class _StressLevel(_IterableAttributes):
    time_s = 'stress_level.stress_level_time[s]'
    value = 'stress_level.stress_level_value'

class _RespirationRate(_IterableAttributes):
    timestamp = 'respiration_rate.timestamp'
    breaths_per_minute = 'respiration_rate.respiration_rate[breaths/min]'

class WellnessColumns:
    # Represents any columns labeled as unknown or unnamed
    invalid = _Invalid()
    file_id = _FileId()
    device_info = _DeviceInfo()
    software = _Software()
    monitoring_info = _MonitoringInfo()
    monitoring = _Monitoring()
    ohr_settings = _OhrSettings()
    event = _Event()
    stress_level = _StressLevel()
    respiration_rate = _RespirationRate()
    monitoring_hr_data = _MonitoringHRData()

class _SleepAssessment:
    timestamp_s = 'sleep_level.timestamp[s]'
    sleep_level = 'sleep_level.sleep_level'
    average_stress = 'sleep_assessment.average_stress_during_sleep'
    combined_awake_score = 'sleep_assessment.combined_awake_score'
    awake_time = 'sleep_assessment.awake_time_score'
    awakenings_count_score = 'sleep_assessment.awakenings_count_score'
    deep_sleep_score = 'sleep_assessment.deep_sleep_score'
    sleep_duration_score = 'sleep_assessment.sleep_duration_score'
    light_sleep_score = 'sleep_assessment.light_sleep_score'
    overall_sleep_score = 'sleep_assessment.overall_sleep_score'
    sleep_quality_score = 'sleep_assessment.sleep_quality_score'
    sleep_recovery_score = 'sleep_assessment.sleep_recovery_score'
    rem_sleep_score = 'sleep_assessment.rem_sleep_score'
    sleep_restlessness_score = 'sleep_assessment.sleep_restlessness_score'
    awakenings_count = 'sleep_assessment.awakenings_count'
    interruptions_score = 'sleep_assessment.interruptions_score'


class SleepColumns:
    invalid = _Invalid()
    file_id = _FileId()
    software = _Software()
    device_info = _DeviceInfo()
    assessment = _SleepAssessment()


class StaticData(_IterableAttributes):
    invalid = _Invalid()
    file_id = _FileId()
    software = _Software()
    device_info = _DeviceInfo()

    def __iter__(self):
        return chain(self.invalid, self.file_id, self.software, self.device_info)
    
    def __len__(self):
        output = 0
        for i in self:
            output += len(i)
        return output