import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const settings = { theme: 'dark', locale: 'auto', experience: 'simple', mode: 'normal', access: 'normal', reduced_motion: false }

function response(value: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(value) } as Response)
}

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <BrowserRouter><App /></BrowserRouter>
    </QueryClientProvider>,
  )
}

describe('GA3BAD workspace', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/')
    vi.stubGlobal('fetch', vi.fn(() => response({
      app: { name: 'GA3BAD', version: 1, local_only: true },
      projects: [],
      settings,
      active_job_id: null,
      queued_jobs: [],
      csrf_token: 'csrf-token',
    })))
  })

  afterEach(() => { cleanup(); vi.unstubAllGlobals() })

  it('renders the warm studio empty state and local project action', async () => {
    renderApp()
    expect(await screen.findByRole('heading', { name: 'GA3BAD Studio' })).toBeInTheDocument()
    expect(screen.getByText('build boldly, review carefully, ship with evidence')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add a local project' })).toBeEnabled()
    expect(screen.getByRole('textbox', { name: 'Workspace path' })).toHaveAttribute('placeholder', 'D:\\projects\\my-app')
  })

  it('renders a task composer, quick actions, and slash command palette', async () => {
    window.history.replaceState({}, '', '/task/thread-1')
    const bootstrap = {
      app: { name: 'GA3BAD', version: 1, local_only: true },
      projects: [{ id: 'project-1', path: 'D:/project', name: 'project', pinned: false, last_opened_at: '', threads: [{ id: 'thread-1', project_id: 'project-1', session_id: 'web-thread-1', title: 'New task', status: 'idle', pinned: false, archived: false, updated_at: '', workflow_mode: 'normal', access_policy: 'default', effective_access: 'normal', model_overrides: {}, state_revision: 1 }] }],
      settings,
      active_job_id: null,
      queued_jobs: [],
      csrf_token: 'csrf-token',
    }
    const snapshot = {
      project: bootstrap.projects[0],
      thread: bootstrap.projects[0].threads[0],
      messages: [{ id: 'message-1', thread_id: 'thread-1', role: 'assistant', content: 'مرحبا بك', technical: false, created_at: '', turn_id: 'turn-1', direction: 'auto' }], jobs: [], settings, presentation: null, dashboard: null,
      turns: [{ id: 'turn-1', messages: [{ id: 'message-1', thread_id: 'thread-1', role: 'assistant', content: 'مرحبا بك', technical: false, created_at: '', turn_id: 'turn-1', direction: 'auto' }], activity: [] }],
      progress: { percent: 0, completed_steps: 0, total_steps: 0, current_step: 'idle', elapsed_seconds: 0, remaining_seconds_low: 0, remaining_seconds_high: 0, estimated_finish_at: null, confidence: 'low', basis: 'baseline', paused_for_attention: false, milestones: [] },
    }
    const telemetry = { ram: { used_bytes: 8_000_000_000, total_bytes: 16_000_000_000, percent: 50 }, gpus: [], cpu: { logical_cores: 8, utilization_percent: null }, context: { used_tokens: 1000, limit_tokens: 16000, remaining_tokens: 15000, percent: 6.25, source: 'latest provider usage' }, sampled_at: '' }
    const advanced = { device: 'auto', context_window: 16384, max_output_tokens: 4096, gpu_layers: -1, cpu_threads: 8, temperature: 0.2, top_p: 0.9, top_k: 40, performance: 'balanced', estimated_minutes_per_step: 30, planning_steps: 16, work_quantum_steps: 24, review_steps: 12, max_provider_retries: 3, ultra_cloud_concurrency: 4, ultra_max_depth: 8 }
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/telemetry')) return response(telemetry)
      if (url.endsWith('/settings/advanced')) return response(advanced)
      return response(url.includes('/threads/thread-1') ? snapshot : bootstrap)
    }))
    renderApp()
    const composer = await screen.findByRole('textbox', { name: 'Task message' })
    fireEvent.change(composer, { target: { value: '/' } })
    expect(await screen.findByText('Refresh durable goal and plan status')).toBeInTheDocument()
    const direction = screen.getByRole('button', { name: 'Use RTL for this output' })
    expect(direction).toHaveTextContent('Auto')
    expect(composer).toHaveAttribute('dir', 'auto')
    expect(screen.getByRole('button', { name: 'Explore workspace files' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: /Default/ }))
    expect(await screen.findByRole('menuitem', { name: /Full Access/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Mode · normal/i }))
    expect(await screen.findByRole('menuitem', { name: /Plan/ })).toBeInTheDocument()
    expect(await screen.findByText('15k free')).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: 'Settings' })[0])
    fireEvent.click(await screen.findByRole('button', { name: 'Advanced' }))
    expect(await screen.findByRole('heading', { name: 'Advanced inference' })).toBeInTheDocument()
    expect(screen.getByText('Context window')).toBeInTheDocument()
  })

  it('keeps planning feedback in Plan mode and exposes long-run progress and Full Access warning', async () => {
    window.history.replaceState({}, '', '/task/thread-plan')
    const thread = { id: 'thread-plan', project_id: 'project-1', session_id: 'web-plan', title: 'Plan task', status: 'awaiting_plan_approval', pinned: false, archived: false, updated_at: '', workflow_mode: 'plan', access_policy: 'default', effective_access: 'normal', model_overrides: {}, state_revision: 3 }
    const bootstrap = { app: { name: 'GA3BAD', version: 1, local_only: true }, projects: [{ id: 'project-1', path: 'D:/project', name: 'project', pinned: false, last_opened_at: '', threads: [thread] }], settings, active_job_id: null, queued_jobs: [], csrf_token: 'csrf-token' }
    const snapshot = {
      project: bootstrap.projects[0], thread, messages: [], turns: [], jobs: [], settings, presentation: null,
      dashboard: { objective: 'Build durable UI', status: 'awaiting_plan_approval', plan_revision: 2, plan_fingerprint: 'f'.repeat(64), approved_revision: null, plan_summary: 'A focused plan', expected_changes: [], tasks: [{ id: 'T1', title: 'Inspect', status: 'done', role: 'architect', acceptance_criteria: [], verification: [], depends_on: [], risk: 'low' }, { id: 'T2', title: 'Implement', status: 'pending', role: 'builder', acceptance_criteria: [], verification: [], depends_on: ['T1'], risk: 'medium' }], activity: [], provider: 'ollama', model: 'local', workspace: 'D:/project' },
      progress: { percent: 50, completed_steps: 1, total_steps: 2, current_step: 'Implement', elapsed_seconds: 3600, remaining_seconds_low: 3600, remaining_seconds_high: 7200, estimated_finish_at: new Date().toISOString(), confidence: 'medium', basis: 'observed milestone velocity', paused_for_attention: true, milestones: [] },
    }
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => response(String(input).includes('/threads/thread-plan') ? snapshot : bootstrap)))
    renderApp()
    expect(await screen.findByRole('button', { name: /Implement plan/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /50%.*Implement/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Keep planning' }))
    expect(screen.getByPlaceholderText(/existing plan will be revised/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Default/ }))
    fireEvent.click(await screen.findByRole('menuitem', { name: /Full Access/ }))
    expect(await screen.findByRole('dialog', { name: 'Full Access warning' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Host Full/ })).toBeInTheDocument()
  })

  it('offers an explicit unsafe kill action for paused work', async () => {
    window.history.replaceState({}, '', '/task/thread-paused')
    const thread = { id: 'thread-paused', project_id: 'project-1', session_id: 'web-paused', title: 'Paused build', status: 'waiting_for_input', pinned: false, archived: false, updated_at: '', workflow_mode: 'normal', access_policy: 'default', effective_access: 'normal', model_overrides: {}, state_revision: 2 }
    const bootstrap = { app: { name: 'GA3BAD', version: 1, local_only: true }, projects: [{ id: 'project-1', path: 'D:/project', name: 'project', pinned: false, last_opened_at: '', threads: [thread] }], settings, active_job_id: null, queued_jobs: [], csrf_token: 'csrf-token' }
    const snapshot = {
      project: bootstrap.projects[0], thread, messages: [], turns: [], settings, dashboard: null, queue: [], draft: { thread_id: thread.id, text: '', revision: 0 },
      jobs: [{ id: 'job-paused', thread_id: thread.id, project_id: 'project-1', status: 'paused', input_text: 'build', created_at: '', cancel_requested: false, blocked_reason: 'attention:decision-1' }],
      presentation: { sequence: 1, mode: 'simple', locale: 'en', transcript: [], activity: { stage: 'paused', summary: 'Waiting', detail: '', running: false, completed: 0, total: 0 }, attention: null, model: 'local', status: 'waiting_for_input', running: false, queued_count: 0, advanced_log: [] },
      progress: { percent: 0, completed_steps: 0, total_steps: 0, current_step: 'waiting', elapsed_seconds: 0, remaining_seconds_low: 0, remaining_seconds_high: 0, estimated_finish_at: null, confidence: 'low', basis: 'paused', paused_for_attention: true, milestones: [] },
      capabilities: { send: { allowed: true, reason: '', remediation: '' }, pause: { allowed: false, reason: 'No action is currently running', remediation: '' }, kill: { allowed: true, reason: '', remediation: '' }, change_mode: { allowed: true, reason: '', remediation: '' } },
    }
    const fetchMock = vi.fn((input: RequestInfo | URL) => response(String(input).includes('/threads/thread-paused') ? snapshot : bootstrap))
    vi.stubGlobal('fetch', fetchMock)
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Kill task immediately' }))
    expect(await screen.findByRole('alertdialog')).toHaveTextContent('does not wait for a safe checkpoint')
    fireEvent.click(screen.getByRole('button', { name: 'Kill task now' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(call => String(call[0]).endsWith('/threads/thread-paused/kill'))).toBe(true))
  })
})
