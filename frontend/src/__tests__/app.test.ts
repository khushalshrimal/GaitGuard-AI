import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GaitGuardApiClient } from '../api/gaitguard';
import { ScreeningResponse } from '../types/api';

describe('GaitGuard AI Phase 17 Frontend Product Logic', () => {
  let client: GaitGuardApiClient;

  beforeEach(() => {
    client = new GaitGuardApiClient('http://localhost:8000');
    vi.restoreAllMocks();
  });

  it('1. verifies health check returns ok status', async () => {
    const mockHealth = { status: 'ok', service: 'gaitguard-api', version: '1.0.0', model_loaded: true };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth
    } as Response);

    const res = await client.getHealth();
    expect(res.status).toBe('ok');
    expect(res.model_loaded).toBe(true);
  });

  it('2. verifies version metadata returns v1 and phase 16/17', async () => {
    const mockVersion = {
      api_version: 'v1',
      model_version: 'bilstm-mode-d-f76',
      pipeline_version: 'phase-12-integrated',
      feature_schema_version: 'schema-76-v1',
      phase: 17
    };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockVersion
    } as Response);

    const res = await client.getVersion();
    expect(res.api_version).toBe('v1');
    expect(res.phase).toBe(17);
  });

  it('3. handles successful screening API response with NORMAL decision', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'success',
      request_id: 'req_test_123',
      inference: {
        raw_probability: 0.10,
        calibrated_probability: 0.12,
        decision: 'NORMAL',
        confidence: 'HIGH',
        threshold: 0.34,
        uncertainty_margin: 0.10
      },
      metadata: {
        request_id: 'req_test_123',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.068
      },
      result_summary: 'Normal gait pattern detected.',
      disclaimer: 'AI-assisted screening tool. Not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'cow_walk.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile, 'COW-101');
    expect(res.status).toBe('success');
    expect(res.inference?.decision).toBe('NORMAL');
    expect(res.inference?.calibrated_probability).toBe(0.12);
  });

  it('4. handles successful screening API response with LAMENESS_RISK decision', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'success',
      request_id: 'req_test_456',
      inference: {
        raw_probability: 0.70,
        calibrated_probability: 0.75,
        decision: 'LAMENESS_RISK',
        confidence: 'HIGH',
        threshold: 0.34,
        uncertainty_margin: 0.10
      },
      metadata: {
        request_id: 'req_test_456',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.069
      },
      result_summary: 'Elevated lameness risk detected.',
      disclaimer: 'AI-assisted screening tool. Not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'lame_cow.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile, 'COW-202');
    expect(res.status).toBe('success');
    expect(res.inference?.decision).toBe('LAMENESS_RISK');
  });

  it('5. handles quality gate RETRY response structure', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'retry',
      request_id: 'req_test_789',
      video_quality: {
        status: 'RETRY',
        quality_score: 35.0,
        keypoint_coverage: 0.40,
        motion_quality: 30.0,
        blur_indicator: 85.0,
        framing_quality: 40.0,
        issues: ['EXCESSIVE_CAMERA_BLUR'],
        user_guidance: ['Ensure stable camera tripod or hand support']
      },
      metadata: {
        request_id: 'req_test_789',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.018
      },
      result_summary: 'Video quality insufficient for gait screening.',
      disclaimer: 'AI-assisted screening tool. Not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'blurry.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile);
    expect(res.status).toBe('retry');
    expect(res.video_quality?.issues).toContain('EXCESSIVE_CAMERA_BLUR');
  });

  it('6. handles INCONCLUSIVE triage decision in uncertainty margin [0.24, 0.44]', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'success',
      request_id: 'req_test_999',
      inference: {
        raw_probability: 0.30,
        calibrated_probability: 0.34,
        decision: 'INCONCLUSIVE',
        confidence: 'LOW',
        threshold: 0.34,
        uncertainty_margin: 0.10
      },
      metadata: {
        request_id: 'req_test_999',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.065
      },
      result_summary: 'Inconclusive screening result within uncertainty margin.',
      disclaimer: 'AI-assisted screening tool. Not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'borderline.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile);
    expect(res.inference?.decision).toBe('INCONCLUSIVE');
  });

  it('7. parses SHAP explanation details when available', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'success',
      request_id: 'req_test_shap',
      inference: {
        raw_probability: 0.70,
        calibrated_probability: 0.75,
        decision: 'LAMENESS_RISK',
        confidence: 'HIGH',
        threshold: 0.34,
        uncertainty_margin: 0.10
      },
      explanation: {
        available: true,
        top_contributors: [
          {
            feature_name: 'back_arch_curvature',
            feature_index: 2,
            attribution: 0.145,
            direction: 'INCREASES_RISK',
            modality: 'biomechanical',
            body_region: 'spine'
          }
        ]
      },
      metadata: {
        request_id: 'req_test_shap',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.070
      },
      result_summary: 'Screening result with SHAP evidence.',
      disclaimer: 'AI-assisted screening tool. Not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'shap_cow.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile);
    expect(res.explanation?.available).toBe(true);
    expect(res.explanation?.top_contributors?.[0].feature_name).toBe('back_arch_curvature');
  });

  it('8. handles server HTTP 400 error cleanly', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Uploaded file content does not match a valid video stream header.' })
    } as Response);

    const dummyFile = new File(['script content'], 'fake.mp4', { type: 'video/mp4' });
    await expect(client.screenVideo(dummyFile)).rejects.toThrow('Uploaded file content does not match a valid video stream header.');
  });

  it('9. handles server HTTP 413 file too large error cleanly', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      statusText: 'Payload Too Large',
      json: async () => ({ detail: 'Uploaded file size exceeds maximum allowed limit of 100 MB.' })
    } as Response);

    const dummyFile = new File(['large content'], 'huge.mp4', { type: 'video/mp4' });
    await expect(client.screenVideo(dummyFile)).rejects.toThrow('Uploaded file size exceeds maximum allowed limit of 100 MB.');
  });

  it('10. verifies non-diagnostic legal disclaimer presence', async () => {
    const mockResponse: ScreeningResponse = {
      status: 'success',
      request_id: 'req_disclaimer_test',
      inference: {
        raw_probability: 0.10,
        calibrated_probability: 0.12,
        decision: 'NORMAL',
        confidence: 'HIGH',
        threshold: 0.34,
        uncertainty_margin: 0.10
      },
      metadata: {
        request_id: 'req_disclaimer_test',
        pipeline_version: 'phase-12-integrated',
        total_processing_time_sec: 0.065
      },
      result_summary: 'Normal gait pattern detected.',
      disclaimer: 'AI-assisted screening tool. This output is not a veterinary diagnosis.'
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    const dummyFile = new File(['header'], 'cow_walk.mp4', { type: 'video/mp4' });
    const res = await client.screenVideo(dummyFile);
    expect(res.disclaimer).toContain('not a veterinary diagnosis');
  });
});
