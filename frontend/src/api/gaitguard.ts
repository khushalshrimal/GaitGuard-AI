import { HealthResponse, VersionResponse, ScreeningResponse } from '../types/api';

const API_BASE_URL = (import.meta as unknown as { env: Record<string, string> }).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export class GaitGuardApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      if (!response.ok) {
        throw new Error(`Health check failed with status ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error('API health check error:', err);
      throw new Error('Unable to connect to the screening service.');
    }
  }

  async getVersion(): Promise<VersionResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/version`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      if (!response.ok) {
        throw new Error(`Version endpoint failed with status ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error('API version error:', err);
      throw new Error('Unable to connect to the screening service.');
    }
  }

  async screenVideo(
    videoFile: File,
    animalId?: string,
    sessionId?: string
  ): Promise<ScreeningResponse> {
    const formData = new FormData();
    formData.append('video', videoFile, videoFile.name);
    
    if (animalId && animalId.trim()) {
      formData.append('animal_id', animalId.trim());
    }
    if (sessionId && sessionId.trim()) {
      formData.append('session_id', sessionId.trim());
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/screen`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorDetail = 'Failed to screen video.';
        try {
          const errData = await response.json();
          if (errData.detail) {
            errorDetail = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch {
          // Fallback to HTTP status text
          errorDetail = `Server error ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorDetail);
      }

      const data: ScreeningResponse = await response.json();
      return data;
    } catch (err: unknown) {
      console.error('API screen error:', err);
      if (err instanceof Error && (err.name === 'TypeError' || err.message.includes('fetch'))) {
        throw new Error('Unable to connect to the screening service. Please ensure backend server is running.');
      }
      throw err;
    }
  }
}

export const api = new GaitGuardApiClient();
