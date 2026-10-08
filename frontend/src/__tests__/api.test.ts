import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GaitGuardApiClient } from '../api/gaitguard';

describe('GaitGuardApiClient', () => {
  let client: GaitGuardApiClient;

  beforeEach(() => {
    client = new GaitGuardApiClient('http://localhost:8000');
    vi.restoreAllMocks();
  });

  it('fetches health check successfully', async () => {
    const mockHealth = { status: 'ok', service: 'gaitguard-api', version: '1.0.0', model_loaded: true };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth
    } as Response);

    const health = await client.getHealth();
    expect(health).toEqual(mockHealth);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/health',
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('fetches version successfully', async () => {
    const mockVersion = {
      api_version: '1.0.0',
      model_version: 'bilstm_v1_calibrated',
      pipeline_version: 'phase_12_v1',
      feature_schema_version: '76_compact_v1',
      phase: 12
    };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockVersion
    } as Response);

    const version = await client.getVersion();
    expect(version).toEqual(mockVersion);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/version',
      expect.objectContaining({ method: 'GET' })
    );
  });

  it('handles API errors gracefully during screening', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ detail: 'Model processing error' })
    } as Response);

    const dummyFile = new File(['dummy content'], 'cow_walk.mp4', { type: 'video/mp4' });

    await expect(client.screenVideo(dummyFile)).rejects.toThrow('Model processing error');
  });
});
