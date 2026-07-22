export type Theme = 'dark' | 'light' | 'system'

export interface Settings {
  theme: Theme
  locale: 'auto' | 'en' | 'ar'
  experience: 'simple' | 'advanced'
  mode: 'plan' | 'normal' | 'ultra'
  access: 'normal' | 'bounded' | 'full'
  reduced_motion: boolean
  inference_profile?: InferenceProfile
}

export interface InferenceProfile {
  device: 'auto' | 'cpu' | 'gpu'
  context_window: number
  max_output_tokens: number
  gpu_layers: number
  cpu_threads: number
  temperature: number
  top_p: number
  top_k: number
  performance: 'eco' | 'balanced' | 'performance'
  estimated_minutes_per_step: number
  planning_steps: number
  work_quantum_steps: number
  review_steps: number
  max_provider_retries: number
  ultra_cloud_concurrency: number
  ultra_max_depth: number
}

export interface TaskProgress {
  percent: number
  completed_steps: number
  total_steps: number
  current_step: string
  elapsed_seconds: number
  remaining_seconds_low: number
  remaining_seconds_high: number
  estimated_finish_at: string | null
  confidence: 'low' | 'medium' | 'high'
  basis: string
  paused_for_attention: boolean
  milestones: Array<{ id: string; title: string; status: string }>
}

export interface ResourceTelemetry {
  ram: { used_bytes: number | null; total_bytes: number | null; percent: number | null }
  gpus: Array<{ index: number; name: string; utilization_percent: number; used_bytes: number; total_bytes: number; percent: number; temperature_c: number }>
  cpu: { logical_cores: number | null; utilization_percent: number | null }
  context: { used_tokens: number | null; limit_tokens: number | null; remaining_tokens: number | null; percent: number | null; source: string }
  sampled_at: string
}

export interface DashboardTask {
  id: string
  title: string
  status: string
  role: string
  acceptance_criteria: string[]
  verification: string[]
  depends_on: string[]
  risk: string
}

export interface Dashboard {
  objective: string
  status: string
  plan_revision: number
  plan_fingerprint?: string
  approved_revision?: number | null
  plan_summary: string
  expected_changes: Array<Record<string, unknown>>
  tasks: DashboardTask[]
  activity: string[]
  provider: string
  model: string
  workspace: string
  [key: string]: unknown
}

export interface Project {
  id: string
  path: string
  name: string
  pinned: boolean
  last_opened_at: string
  threads?: ThreadSummary[]
}

export interface ThreadSummary {
  id: string
  project_id: string
  session_id: string
  goal_id?: string | null
  title: string
  status: string
  pinned: boolean
  archived: boolean
  updated_at: string
  workflow_mode: 'plan' | 'normal' | 'ultra'
  access_policy: 'default' | 'bounded' | 'full'
  effective_access?: 'normal' | 'bounded' | 'full' | 'host'
  model_overrides: Record<string, string>
  state_revision: number
  view_mode: 'transcript' | 'visualize'
  plan_series_id: string
  pending_model_overrides: Record<string, string>
}

export interface Message {
  id: number
  role: 'user' | 'assistant' | string
  content: string
  technical: boolean
  created_at: string
  turn_id: string
  direction: 'auto' | 'rtl'
}

export interface TurnActivity {
  id: number
  turn_id: string
  kind: string
  summary: string
  details: string
  created_at: string
}

export interface ConversationTurn {
  id: string
  messages: Message[]
  activity: TurnActivity[]
  job?: Job | null
}

export interface ModelDescriptor {
  id: string
  provider: string
  model: string
  execution_class: 'local' | 'cloud'
  host?: string | null
  capabilities: string[]
  label?: string | null
  source: string
}

export interface ModelSettings {
  models: ModelDescriptor[]
  defaults: Record<string, string>
  overrides: Record<string, string>
  diagnostics: Array<{ source: string; message: string }>
  roles: Array<'main' | 'router' | 'verifier' | 'embedding'>
  embedding_default?: string
}

export interface AttentionOption {
  key: string
  label: string
  value: string
  description?: string
  shortcut?: string
  primary?: boolean
}

export interface Attention {
  id: string
  kind: string
  title: string
  message?: string
  details?: string
  allow_custom?: boolean
  options: AttentionOption[]
}

export interface Presentation {
  mode: string
  locale: string
  activity: {
    stage: string
    summary: string
    completed: number
    total: number
    last_success?: string
  }
  attention: Attention | null
  model: string
  status: string
  running: boolean
  queued_count: number
  advanced_log: string[]
}

export interface Job {
  id: string
  thread_id: string
  project_id: string
  status: string
  input_text: string
  created_at: string
  completed_at?: string | null
  cancel_requested: boolean
  kind?: string
  delivery?: 'queue' | 'guidance'
  blocked_reason?: string
}

export interface ActionCapability {
  allowed: boolean
  reason: string
  remediation: string
}

export interface ThreadDraft {
  thread_id: string
  text: string
  revision: number
  updated_at?: string | null
}

export interface VisualizationNode {
  id: string
  kind: string
  label: string
  summary: string
  status: string
  parent_id?: string | null
  details?: Record<string, unknown>
}

export interface VisualizationEdge {
  id: string
  source: string
  target: string
  kind: string
}

export interface VisualizationSnapshot {
  version: 1
  thread_id: string
  mode: 'plan' | 'normal' | 'ultra'
  revision: number
  current_node_id: string
  nodes: VisualizationNode[]
  edges: VisualizationEdge[]
  updated_at: string
}

export interface TerminalSession {
  id: string
  thread_id: string
  mode: 'managed' | 'docker' | 'host'
  status: string
  cwd: string
  history: string[]
  scrollback: string
}

export interface ThreadSnapshot {
  project: Project
  thread: ThreadSummary
  messages: Message[]
  turns: ConversationTurn[]
  jobs: Job[]
  settings: Settings
  presentation: Presentation | null
  dashboard: Dashboard | null
  progress: TaskProgress
  queue: Job[]
  draft: ThreadDraft
  capabilities: Record<string, ActionCapability>
}

export interface Bootstrap {
  app: { name: string; version: number; local_only: boolean }
  projects: Project[]
  settings: Settings
  active_job_id: string | null
  queued_jobs: Job[]
  csrf_token: string
}

export interface WebEvent {
  version: 1
  sequence: number
  project_id?: string | null
  thread_id?: string | null
  type: string
  payload: Record<string, unknown>
  emitted_at: string
}
