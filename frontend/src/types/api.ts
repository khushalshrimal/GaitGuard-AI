export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
}

export interface VersionResponse {
  api_version: string;
  model_version: string;
  pipeline_version: string;
  feature_schema_version: string;
  phase: number;
}

export interface QualityDetails {
  status: 'READY' | 'RETRY' | 'INCONCLUSIVE';
  quality_score: number;
  keypoint_coverage: number;
  motion_quality: number;
  blur_indicator: number;
  framing_quality: number;
  issues: string[];
  user_guidance: string[];
}

export interface InferenceDetails {
  raw_probability: number;
  calibrated_probability: number;
  decision: 'NORMAL' | 'LAMENESS_RISK' | 'INCONCLUSIVE';
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  threshold: number;
  uncertainty_margin: number;
}

export interface TopContributor {
  feature_name: string;
  feature_index: number;
  attribution: number;
  direction: 'INCREASES_RISK' | 'DECREASES_RISK' | 'LOW_CONTRIBUTION';
  modality: string;
  body_region: string;
}

export interface DerivedGaitEvidenceItem {
  name: string;
  value: number;
  relative_contribution: number;
  direction: string;
  interpretation: string;
}

export interface ExplanationDetails {
  available: boolean;
  top_contributors?: TopContributor[];
  modality_attribution?: Record<string, number>;
  body_region_attribution?: Record<string, number>;
  temporal_attribution?: Record<string, number>;
  derived_gait_evidence?: DerivedGaitEvidenceItem[];
  reason?: string;
}

export interface MetadataDetails {
  request_id: string;
  animal_id?: string | null;
  session_id?: string | null;
  pipeline_version: string;
  total_processing_time_sec: number;
}

export interface ScreeningResponse {
  status: 'success' | 'retry' | 'error';
  request_id: string;
  video_quality?: QualityDetails | null;
  inference?: InferenceDetails | null;
  explanation?: ExplanationDetails | null;
  metadata: MetadataDetails;
  result_summary: string;
  disclaimer: string;
}
