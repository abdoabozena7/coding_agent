import { z } from 'zod'
import type { Bootstrap, InferenceProfile, Message, ModelSettings, ResourceTelemetry, Settings, TerminalSession, ThreadDraft, ThreadSnapshot, VisualizationSnapshot, WebEvent } from './types'

let csrfToken = ''

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details: Record<string, unknown> = {}) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set('content-type', 'application/json')
  if (init.method && !['GET', 'HEAD'].includes(init.method.toUpperCase())) {
    headers.set('x-ga3bad-csrf', csrfToken)
  }
  const response = await fetch(path, { ...init, headers, credentials: 'same-origin' })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    let code = 'http_error'
    let details: Record<string, unknown> = {}
    try {
      const value = await response.json()
      message = value.detail?.message ?? value.message ?? message
      code = value.detail?.code ?? value.code ?? code
      details = value.detail?.details ?? value.details ?? details
    } catch {
      // Preserve the HTTP status when an intermediary returns text.
    }
    throw new ApiError(response.status, code, message, details)
  }
  return response.json() as Promise<T>
}

export async function getBootstrap(): Promise<Bootstrap> {
  const value = await request<Bootstrap>('/api/v1/bootstrap')
  csrfToken = value.csrf_token
  return value
}

export const api = {
  addProject: (path: string) => request<{ project: unknown; created: boolean }>('/api/v1/projects', {
    method: 'POST', body: JSON.stringify({ path }),
  }),
  createThread: (projectId: string, title = 'New task') => request(`/api/v1/projects/${projectId}/threads`, {
    method: 'POST', body: JSON.stringify({ title }),
  }),
  getThread: (threadId: string) => request<ThreadSnapshot>(`/api/v1/threads/${threadId}`),
  patchThread: (threadId: string, body: Record<string, unknown>) => request(`/api/v1/threads/${threadId}`, {
    method: 'PATCH', body: JSON.stringify(body),
  }),
  setMode: (threadId: string, mode: 'plan' | 'normal' | 'ultra', expectedRevision: number) => request(
    `/api/v1/threads/${threadId}/mode`,
    { method: 'POST', body: JSON.stringify({ mode, expected_revision: expectedRevision }) },
  ),
  setMessageDirection: (threadId: string, messageId: number, direction: 'auto' | 'rtl') => request<Message>(
    `/api/v1/threads/${threadId}/messages/${messageId}`,
    { method: 'PATCH', body: JSON.stringify({ direction }) },
  ),
  planDecision: (
    threadId: string,
    body: { action: 'implement' | 'keep_planning'; revision: number; fingerprint: string; feedback?: string },
  ) => request(`/api/v1/threads/${threadId}/plan/decision`, {
    method: 'POST',
    body: JSON.stringify({ ...body, feedback: body.feedback ?? '', client_request_id: crypto.randomUUID() }),
  }),
  changeAccess: (
    threadId: string,
    policy: 'default' | 'bounded' | 'full' | 'host',
    expectedRevision: number,
    confirmationToken = '',
  ) => request<Record<string, unknown>>(`/api/v1/threads/${threadId}/access`, {
    method: 'POST',
    body: JSON.stringify({ policy, expected_revision: expectedRevision, confirmation_token: confirmationToken }),
  }),
  submit: (threadId: string, text: string, delivery: 'queue' | 'guidance' = 'queue') => request(`/api/v1/threads/${threadId}/inputs`, {
    method: 'POST',
    body: JSON.stringify({
      kind: text.trimStart().startsWith('/') ? 'command' : 'message',
      text,
      delivery,
      client_request_id: crypto.randomUUID(),
    }),
  }),
  resolveAttention: (threadId: string, requestId: string, optionKey: string, text = '') => request(
    `/api/v1/threads/${threadId}/attention/${requestId}/resolve`,
    { method: 'POST', body: JSON.stringify({ option_key: optionKey, text }) },
  ),
  checkpoint: (jobId: string) => request(`/api/v1/jobs/${jobId}/checkpoint`, { method: 'POST' }),
  killTask: (threadId: string) => request(`/api/v1/threads/${threadId}/kill`, { method: 'POST' }),
  resume: (threadId: string) => request(`/api/v1/threads/${threadId}/resume`, { method: 'POST' }),
  queue: (threadId: string) => request<{ items: unknown[]; limit: number }>(`/api/v1/threads/${threadId}/queue`),
  cancelQueued: (threadId: string, jobId: string) => request(`/api/v1/threads/${threadId}/queue/${jobId}`, { method: 'DELETE' }),
  continueQueue: (threadId: string) => request(`/api/v1/threads/${threadId}/queue/continue`, { method: 'POST' }),
  saveDraft: (threadId: string, text: string, expectedRevision?: number) => request<ThreadDraft>(`/api/v1/threads/${threadId}/draft`, {
    method: 'PATCH', body: JSON.stringify({ text, expected_revision: expectedRevision }),
  }),
  setView: (threadId: string, viewMode: 'transcript' | 'visualize', expectedRevision: number) => request(
    `/api/v1/threads/${threadId}/view`,
    { method: 'PATCH', body: JSON.stringify({ view_mode: viewMode, expected_revision: expectedRevision }) },
  ),
  visualization: (threadId: string) => request<VisualizationSnapshot>(`/api/v1/threads/${threadId}/visualization`),
  pickFolder: () => request<{ path: string; cancelled: boolean }>('/api/v1/projects/pick-folder', { method: 'POST' }),
  openTerminal: (threadId: string) => request<TerminalSession>(`/api/v1/threads/${threadId}/terminal`, { method: 'POST' }),
  terminalCommand: (sessionId: string, command: string) => request<TerminalSession & { output: string; returncode: number }>(
    `/api/v1/terminal/${sessionId}/command`, { method: 'POST', body: JSON.stringify({ command }) },
  ),
  closeTerminal: (sessionId: string) => request(`/api/v1/terminal/${sessionId}`, { method: 'DELETE' }),
  patchSettings: (body: Partial<Settings>) => request<Settings>('/api/v1/settings', {
    method: 'PATCH', body: JSON.stringify(body),
  }),
  advancedSettings: () => request<InferenceProfile>('/api/v1/settings/advanced'),
  patchAdvancedSettings: (body: InferenceProfile) => request<InferenceProfile>('/api/v1/settings/advanced', {
    method: 'PATCH', body: JSON.stringify(body),
  }),
  modelSettings: (threadId?: string) => request<ModelSettings>(
    `/api/v1/model-settings${threadId ? `?thread_id=${encodeURIComponent(threadId)}` : ''}`,
  ),
  refreshModels: () => request<ModelSettings>('/api/v1/models/refresh', { method: 'POST' }),
  validateModel: (descriptorId: string) => request('/api/v1/models/validate', {
    method: 'POST', body: JSON.stringify({ descriptor_id: descriptorId }),
  }),
  setDefaultModelRole: (role: string, descriptorId: string) => request<ModelSettings>(
    `/api/v1/model-settings/${role}`,
    { method: 'PATCH', body: JSON.stringify({ descriptor_id: descriptorId }) },
  ),
  setThreadModelRole: (threadId: string, role: string, descriptorId: string) => request(
    `/api/v1/threads/${threadId}/models/${role}`,
    { method: 'PATCH', body: JSON.stringify({ descriptor_id: descriptorId }) },
  ),
  dockerStatus: () => request<Record<string, unknown>>('/api/v1/execution/docker'),
  setupDocker: () => request<Record<string, unknown>>('/api/v1/execution/docker/setup', { method: 'POST' }),
  telemetry: (threadId: string) => request<ResourceTelemetry>(`/api/v1/threads/${threadId}/telemetry`),
  filePreview: (threadId: string, path: string) => request<{ path: string; size: number; content: string; truncated: boolean }>(
    `/api/v1/threads/${threadId}/files/preview?path=${encodeURIComponent(path)}`,
  ),
  inspector: (threadId: string, name: string) => request<Record<string, unknown>>(
    `/api/v1/threads/${threadId}/${name}`,
  ),
}

const eventSchema = z.object({
  version: z.literal(1),
  sequence: z.number(),
  project_id: z.string().nullable().optional(),
  thread_id: z.string().nullable().optional(),
  type: z.string(),
  payload: z.record(z.string(), z.unknown()),
  emitted_at: z.string(),
})

export function connectEvents(onEvent: (event: WebEvent) => void, onState: (online: boolean) => void, onGap?: () => void) {
  let socket: WebSocket | null = null
  let closed = false
  let retry = 500
  let lastSequence: number | null = null

  const connect = () => {
    if (closed) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${location.host}/api/v1/ws`)
    socket.onopen = () => { retry = 500; onState(true) }
    socket.onmessage = (message) => {
      try {
        const parsed = eventSchema.safeParse(JSON.parse(String(message.data)))
        if (parsed.success) {
          const event = parsed.data as WebEvent
          if (event.type === 'app.snapshot') lastSequence = event.sequence
          else {
            if (lastSequence !== null && event.sequence !== lastSequence + 1) onGap?.()
            lastSequence = event.sequence
          }
          onEvent(event)
        }
      } catch {
        // A malformed transient event is ignored; REST snapshots remain canonical.
      }
    }
    socket.onclose = () => {
      onState(false)
      if (!closed) {
        window.setTimeout(connect, retry)
        retry = Math.min(8_000, retry * 1.8)
      }
    }
  }
  connect()
  return () => { closed = true; socket?.close() }
}
