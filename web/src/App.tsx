import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlignRight, Archive, Bot, Check, ChevronDown, ChevronRight, Circle, CircleAlert, CircleCheck, Clock3, Cpu,
  CircleHelp, Clipboard, Container, Eye, FileDiff, Folder, FolderOpen, FolderTree,
  GitBranch, Lightbulb, ListTodo, ListTree, LoaderCircle, Menu, Monitor, Moon,
  PanelLeftClose, PanelRight, Pin, Play, Plus, RefreshCw, RotateCcw, Search, Send,
  Gauge, MemoryStick, Server, Settings as SettingsIcon, Shield, Sparkles, Square, SquarePlus,
  SquareTerminal, Sun, X, Wifi, WifiOff, Network, Pause, ListPlus, MessageSquarePlus, Trash2, OctagonX,
} from 'lucide-react'
import { api, connectEvents, getBootstrap } from './api'
import type { ActionCapability, Attention, Bootstrap, ConversationTurn, Dashboard, DashboardTask, InferenceProfile, Job, Message, ModelSettings, Presentation, ResourceTelemetry, Settings, TaskProgress, TerminalSession, ThreadSnapshot, ThreadSummary, VisualizationSnapshot } from './types'
import '@xterm/xterm/css/xterm.css'

const VisualizeView = lazy(() => import('./VisualizeView').then(module => ({ default: module.VisualizeView })))

const commands = [
  ['/status', 'Refresh durable goal and plan status'],
  ['/plan', 'Open the complete approval-bound plan'],
  ['/auto', 'Continue until completion or a real decision'],
  ['/pause', 'Checkpoint the current task safely'],
  ['/resume', 'Continue a paused task'],
  ['/diff', 'Review current workspace changes'],
  ['/history', 'Show durable events'],
  ['/agents', 'Inspect active specialist agents'],
  ['/memory', 'Inspect Project Brain'],
  ['/thinking', 'Show redacted session thoughts'],
  ['/permissions ', 'Choose normal or full access'],
  ['/mode ', 'Choose Normal or Ultra'],
  ['/model ', 'Switch model at a safe checkpoint'],
] as const

const inspectors = [
  ['plan', 'Plan'], ['changes', 'Changes'], ['agents', 'Agents'],
  ['versions', 'Versions'], ['history', 'History'], ['evidence', 'Evidence'],
  ['artifacts', 'Artifacts'], ['memory', 'Memory'], ['traces', 'Trace'],
  ['metrics', 'Metrics'], ['resources', 'Processes'], ['files', 'Files'],
] as const

function taskPath(threadId: string) {
  return `/task/${encodeURIComponent(threadId)}`
}

function useSelectedThreadId() {
  const location = useLocation()
  const match = location.pathname.match(/^\/task\/([^/]+)/)
  return match ? decodeURIComponent(match[1]) : null
}

function statusLabel(status?: string) {
  return String(status || 'idle').replaceAll('_', ' ')
}

function ProjectMark() {
  return (
    <span className="project-mark" aria-hidden="true">
      {Array.from({ length: 9 }, (_, index) => <i key={index} />)}
    </span>
  )
}

function IconButton({ label, children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return <button className="icon-button" aria-label={label} title={label} {...props}>{children}</button>
}

function ToolButton({ label, active, children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return <button className={`tool-button ${active ? 'active' : ''}`} aria-label={label} data-tooltip={label} {...props}>{children}</button>
}

function ProjectSidebar({
  bootstrap,
  selectedThreadId,
  open,
  onClose,
  onSelect,
  onNewThread,
  onAddProject,
  onPatchThread,
  onSettings,
}: {
  bootstrap: Bootstrap
  selectedThreadId: string | null
  open: boolean
  onClose: () => void
  onSelect: (id: string) => void
  onNewThread: (projectId?: string) => void
  onAddProject: (path?: string) => void
  onPatchThread: (threadId: string, patch: Record<string, unknown>) => void
  onSettings: () => void
}) {
  const [search, setSearch] = useState('')
  const [workspacePath, setWorkspacePath] = useState('')
  const [archived, setArchived] = useState(false)
  const projects = bootstrap.projects
  const defaultProject = projects.find(project => project.threads?.some(t => t.id === selectedThreadId)) ?? projects[0]
  return (
    <aside className={`sidebar ${open ? 'is-open' : ''}`} aria-label="Projects and tasks">
      <div className="brand-row">
        <div className="brand"><ProjectMark /><span>GA3BAD</span></div>
        <IconButton label="Close sidebar" onClick={onClose}><PanelLeftClose size={17} /></IconButton>
      </div>
      <button className="new-task" onClick={() => onNewThread(defaultProject?.id)}>
        <SquarePlus size={17} /><span>New chat</span>
      </button>
      <button className="new-project" onClick={() => onAddProject()}>
        <FolderOpen size={17} /><span>New project</span>
      </button>
      {!projects.length ? (
        <form className="workspace-entry" onSubmit={event => { event.preventDefault(); if (workspacePath.trim()) onAddProject(workspacePath) }}>
          <input value={workspacePath} onChange={event => setWorkspacePath(event.target.value)} placeholder="D:\projects\my-app" aria-label="Workspace path" />
          <button type="button" onClick={() => onAddProject()} aria-label="Choose workspace folder"><FolderOpen size={17} /></button>
          <button type="submit" disabled={!workspacePath.trim()} aria-label="Add typed workspace path"><Send size={15} /></button>
        </form>
      ) : (
        <label className="sidebar-search">
          <Search size={15} />
          <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search tasks" />
        </label>
      )}
      <div className="projects-label">
        <span>Projects</span><ChevronDown size={13} />
        <button onClick={() => onAddProject()} aria-label="Add project"><SquarePlus size={14} /></button>
      </div>
      <div className="project-scroll">
        {projects.map(project => {
          const threads = (project.threads ?? []).filter(thread => {
            if (thread.archived !== archived) return false
            return !search || thread.title.toLowerCase().includes(search.toLowerCase())
          })
          return (
            <section className="project-group" key={project.id}>
              <div className="project-heading" title={project.path}>
                <Folder size={15} /><span>{project.name}</span>
                <button onClick={() => onNewThread(project.id)} aria-label={`New task in ${project.name}`}><Plus size={14} /></button>
              </div>
              <div className="thread-list">
                {threads.map(thread => (
                  <motion.div layout key={thread.id} className={`thread-row ${selectedThreadId === thread.id ? 'selected' : ''}`}>
                    <button className="thread-main" onClick={() => onSelect(thread.id)}>
                      <span className={`status-dot status-${thread.status}`} />
                      <span className="thread-title">{thread.title}</span>
                    </button>
                    <div className="thread-actions">
                      <IconButton
                        label={thread.pinned ? 'Unpin task' : 'Pin task'}
                        onClick={() => onPatchThread(thread.id, { pinned: !thread.pinned })}
                      ><Pin size={13} fill={thread.pinned ? 'currentColor' : 'none'} /></IconButton>
                      <IconButton
                        label={thread.archived ? 'Restore task' : 'Archive task'}
                        onClick={() => onPatchThread(thread.id, { archived: !thread.archived })}
                      ><Archive size={13} /></IconButton>
                    </div>
                  </motion.div>
                ))}
                {!threads.length && <p className="empty-list">{archived ? 'No archived tasks' : 'No tasks yet'}</p>}
              </div>
            </section>
          )
        })}
        {!projects.length && <p className="empty-workspace-copy">Your active workspace will appear here.</p>}
      </div>
      <div className="sidebar-footer">
        <button onClick={() => setArchived(value => !value)}><Archive size={15} />{archived ? 'Recent tasks' : 'Archived'}</button>
        <button onClick={onSettings}><SettingsIcon size={15} />Settings</button>
      </div>
    </aside>
  )
}

function EmptyWorkspace({ projectName }: { projectName?: string }) {
  return (
    <motion.div className="empty-workspace" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
      <span className="empty-eyebrow">{projectName ? `${projectName} is ready` : 'Open a workspace to begin'}</span>
      <ProjectMark />
      <h1>GA3BAD Studio</h1>
      <p className="studio-tagline">build boldly, review carefully, ship with evidence</p>
    </motion.div>
  )
}

function MessageView({
  message,
  onDirection,
}: {
  message: Message
  onDirection?: (message: Message, direction: 'auto' | 'rtl') => void
}) {
  if (message.role === 'user') {
    return (
      <motion.article id={`message-${message.id}`} className="message user-message" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <div className="message-label">You</div>
        <div className="user-bubble">{message.content}</div>
      </motion.article>
    )
  }
  return (
    <motion.article id={`message-${message.id}`} className={`message assistant-message ${message.technical ? 'technical' : ''}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="message-label"><span><ProjectMark /> GA3BAD</span>{onDirection && <button className={message.direction === 'rtl' ? 'active' : ''} aria-label={message.direction === 'rtl' ? 'Use automatic direction for this output' : 'Use RTL for this output'} onClick={() => onDirection(message, message.direction === 'rtl' ? 'auto' : 'rtl')}><AlignRight size={14} /><span>{message.direction === 'rtl' ? 'RTL' : 'Auto'}</span></button>}</div>
      <div className={`markdown output-direction-${message.direction}`} dir={message.direction === 'rtl' ? 'rtl' : 'auto'}><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>
    </motion.article>
  )
}

function TurnActivity({ turn, onReview, onUndo }: { turn: ConversationTurn; onReview: () => void; onUndo: () => void }) {
  const running = turn.job?.status === 'running'
  const canUndo = turn.activity?.some(item => item.kind === 'checkpoint' && item.details.includes('"valid": true'))
  const legacyLabels: Record<string, string> = {
    'planning.inspection_recorded': 'Workspace inspection recorded',
    'goal_contract.projected': 'Execution brief prepared',
    'provider.capability_selected': 'Model capabilities verified',
    action_started: 'Safe action started',
    action_completed: 'Safe action completed',
  }
  const activity = (turn.activity ?? []).flatMap(item => {
    if (item.kind === 'model_thought' || item.kind === 'model_text') return []
    const summary = legacyLabels[item.kind]
      ?? (/^action_[a-f0-9]+$/i.test(item.summary.trim()) ? 'Safe action started' : item.summary)
    return [{ ...item, summary }]
  })
  const canReview = activity.some(item => ['edit', 'checkpoint', 'changes'].includes(item.kind))
  if (!activity.length && !turn.job) return null
  return (
    <details className="turn-activity" open={running || undefined}>
      <summary><span><SquareTerminal size={14} />{running ? 'Working' : 'Work details'}</span><small>{activity.length} events</small><ChevronRight size={14} /></summary>
      <div className="turn-activity-scroll">
        {activity.slice(-16).map(item => <div className={`turn-event turn-event-${item.kind}`} key={item.id}><CircleCheck size={13} /><span>{item.summary}</span></div>)}
      </div>
      {!running && (canReview || canUndo) && <div className="turn-result-actions">{canReview && <button onClick={onReview}><Eye size={14} />Review changes</button>}{canUndo && <button onClick={onUndo}><RotateCcw size={14} />Undo</button>}</div>}
    </details>
  )
}

function MessageNavigator({ turns }: { turns: ConversationTurn[] }) {
  const [open, setOpen] = useState(false)
  const prompts = turns.flatMap(turn => {
    const message = turn.messages.find(item => item.role === 'user')
    return message ? [{ turn, message }] : []
  })
  if (prompts.length < 2) return null
  return (
    <aside className="message-navigator" aria-label="User message navigator">
      <span className="message-navigator-label">Prompts</span>
      <button className="message-navigator-toggle" aria-expanded={open} onClick={() => setOpen(value => !value)}><ListTree size={14} />Prompts <i>{prompts.length}</i></button>
      <div className={open ? 'open' : ''}>{prompts.map(({ message }, index) => <button key={message.id} title={message.content} onClick={() => { document.getElementById(`message-${message.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); setOpen(false) }}><i>{index + 1}</i><span>{message.content}</span></button>)}</div>
    </aside>
  )
}

function formatDuration(milliseconds: number) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000))
  const minutes = Math.floor(seconds / 60)
  return minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`
}

function Elapsed({ since }: { since?: string }) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!since) return
    const timer = window.setInterval(() => setNow(Date.now()), 1_000)
    return () => window.clearInterval(timer)
  }, [since])
  if (!since) return null
  return <span className="worked-for">Worked for {formatDuration(now - new Date(since).getTime())}</span>
}

function ActivityTimeline({ presentation, startedAt }: { presentation: Presentation | null; startedAt?: string }) {
  if (!presentation?.running) return null
  return (
    <section className="activity-timeline" aria-label="Work activity">
      <Elapsed since={startedAt} />
      <div className="activity-row current"><LoaderCircle size={14} /><span>{presentation.activity.summary || 'Working at the next safe checkpoint'}</span></div>
    </section>
  )
}

function PlanCard({ dashboard }: { dashboard: Dashboard }) {
  const [expanded, setExpanded] = useState(false)
  const planText = [dashboard.objective, dashboard.plan_summary].filter(Boolean).join('\n\n')
  return (
    <motion.section className={`plan-card ${expanded ? 'expanded' : 'collapsed'}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div className="plan-card-header">
        <span><Lightbulb size={15} />Plan</span>
        <IconButton label="Copy plan" onClick={() => void navigator.clipboard?.writeText(planText)}><Clipboard size={14} /></IconButton>
      </div>
      <h2>{dashboard.objective}</h2>
      {expanded && dashboard.plan_summary && <div className="plan-summary"><ReactMarkdown remarkPlugins={[remarkGfm]}>{dashboard.plan_summary}</ReactMarkdown></div>}
      {!!dashboard.tasks.length && (
        <ol className="plan-card-tasks">
          {dashboard.tasks.slice(0, expanded ? dashboard.tasks.length : 3).map(task => <li key={task.id}>{task.title}</li>)}
        </ol>
      )}
      <button className="plan-expand" onClick={() => setExpanded(value => !value)}>{expanded ? 'Collapse plan' : `Expand plan${dashboard.tasks.length > 3 ? ` · ${dashboard.tasks.length} steps` : ''}`}<ChevronDown size={14} /></button>
    </motion.section>
  )
}

function PlanDecisionPanel({
  dashboard,
  busy,
  blockedReason,
  onImplement,
  onRevise,
}: {
  dashboard: Dashboard
  busy: boolean
  blockedReason?: string
  onImplement: () => void
  onRevise: (feedback: string) => void
}) {
  const [revising, setRevising] = useState(false)
  const [feedback, setFeedback] = useState('')
  return (
    <motion.section className="plan-decision" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div><span className="eyebrow">Plan r{dashboard.plan_revision} is ready</span><strong>Review the latest revision before tools can run.</strong></div>
      {revising ? (
        <div className="plan-feedback"><textarea autoFocus value={feedback} onChange={event => setFeedback(event.target.value)} placeholder="Tell GA3BAD what to add or change. The existing plan will be revised, not restarted." /><button disabled={busy || !feedback.trim()} onClick={() => onRevise(feedback)}><ListTodo size={15} />Revise plan</button><button onClick={() => setRevising(false)}>Back</button></div>
      ) : (
        <div className="plan-decision-actions"><button className="implement-plan" disabled={busy} title={blockedReason} onClick={onImplement}><Play size={15} fill="currentColor" />Implement plan</button><button disabled={busy} title={blockedReason} onClick={() => setRevising(true)}>Keep planning</button>{blockedReason && <small>{blockedReason}</small>}</div>
      )}
    </motion.section>
  )
}

function QueueTray({ items, onCancel }: { items: Job[]; onCancel: (id: string) => void }) {
  const [open, setOpen] = useState(false)
  if (!items.length) return null
  return (
    <div className="queue-tray">
      <button className="queue-summary" onClick={() => setOpen(value => !value)} aria-expanded={open}><ListTodo size={14} /><span>{items.length}/10 queued</span><small>{items[0]?.input_text}</small><ChevronDown size={13} /></button>
      <AnimatePresence>{open && <motion.div className="queue-list" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}>{items.map((item, index) => <div key={item.id}><i>{index + 1}</i><span>{item.input_text}</span><button aria-label={`Cancel queued message ${index + 1}`} onClick={() => onCancel(item.id)}><Trash2 size={13} /></button></div>)}</motion.div>}</AnimatePresence>
    </div>
  )
}

function diffStats(value: unknown) {
  const diff = String((value as { diff?: unknown } | undefined)?.diff ?? '')
  const files = new Set(Array.from(diff.matchAll(/^diff --git a\/(.+?) b\/(.+)$/gm), match => match[2])).size
  const additions = diff.split('\n').filter(line => line.startsWith('+') && !line.startsWith('+++')).length
  const deletions = diff.split('\n').filter(line => line.startsWith('-') && !line.startsWith('---')).length
  return { files, additions, deletions }
}

function TaskStatusIcon({ task }: { task: DashboardTask }) {
  const status = task.status.toLowerCase()
  if (['done', 'completed', 'skipped'].includes(status)) return <CircleCheck size={15} />
  if (['running', 'in_progress'].includes(status)) return <LoaderCircle className="spin" size={15} />
  return <Circle size={15} />
}

function durationLabel(seconds: number) {
  const minutes = Math.max(0, Math.round(seconds / 60))
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest ? `${hours}h ${rest}m` : `${hours}h`
}

function bytesLabel(value: number | null | undefined) {
  if (value == null) return 'N/A'
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(0)} MB`
  return `${(value / 1024 ** 3).toFixed(1)} GB`
}

function PlanProgress({ dashboard, progress, changes }: { dashboard: Dashboard; progress: TaskProgress; changes?: unknown }) {
  const [open, setOpen] = useState(false)
  const tasks = dashboard.tasks ?? []
  const completed = tasks.filter(task => ['done', 'completed', 'skipped'].includes(task.status.toLowerCase())).length
  const stats = diffStats(changes)
  const eta = progress.paused_for_attention
    ? 'Waiting for your decision'
    : `~${durationLabel(progress.remaining_seconds_low)}–${durationLabel(progress.remaining_seconds_high)} left`
  return (
    <div className="plan-progress">
      <AnimatePresence>
        {open && (
          <motion.div className="plan-progress-popover" initial={{ opacity: 0, y: 8, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 6, scale: .98 }}>
            <div className="progress-estimate-copy"><strong>{progress.percent}% complete</strong><span>{eta}</span><small>{progress.confidence} confidence · {progress.basis}</small>{progress.estimated_finish_at && <small>Expected around {new Date(progress.estimated_finish_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</small>}</div>
            <div className="progress-track" aria-label={`${progress.percent}% complete`}><span style={{ width: `${progress.percent}%` }} /></div>
            {tasks.map(task => <div className="plan-progress-task" key={task.id}><TaskStatusIcon task={task} /><span>{task.title}</span></div>)}
          </motion.div>
        )}
      </AnimatePresence>
      <button className="progress-flow" aria-expanded={open} onClick={() => setOpen(value => !value)}>
        <span className="progress-ring" style={{ '--progress': `${progress.percent * 3.6}deg` } as React.CSSProperties}>{completed < tasks.length ? <span>{progress.percent}%</span> : <CircleCheck size={16} />}</span>
        <span className="progress-now"><strong>{progress.current_step || `Step ${completed + 1}`}</strong><small>Step {completed} / {tasks.length} · {eta}</small></span>
        <span className="progress-mini-track"><i style={{ width: `${progress.percent}%` }} /></span>
        <span className="progress-changes">{stats.files ? `${stats.files} files` : `${dashboard.expected_changes?.length || 0} planned`}{(stats.additions > 0 || stats.deletions > 0) && <small><b>+{stats.additions}</b> <em>-{stats.deletions}</em></small>}</span>
      </button>
    </div>
  )
}

function ResourceStrip({ threadId }: { threadId: string | null }) {
  const query = useQuery({
    queryKey: ['telemetry', threadId],
    queryFn: () => api.telemetry(threadId!),
    enabled: Boolean(threadId),
    refetchInterval: 4_000,
  })
  const value = query.data
  const gpu = value?.gpus?.[0]
  const items: Array<{ label: string; value: string; percent: number | null; title: string; always?: boolean }> = [
    { label: 'CPU', value: value?.cpu?.utilization_percent == null ? '—' : `${value.cpu.utilization_percent}%`, percent: value?.cpu?.utilization_percent ?? null, title: value?.cpu?.logical_cores ? `System-wide CPU · ${value.cpu.logical_cores} logical cores · sampled ${value.sampled_at}` : 'CPU telemetry unavailable' },
    { label: 'RAM', value: value?.ram?.percent == null ? '—' : `${value.ram.percent}%`, percent: value?.ram?.percent ?? null, title: value?.ram ? `System-wide memory · ${bytesLabel(value.ram.used_bytes)} of ${bytesLabel(value.ram.total_bytes)} · sampled ${value.sampled_at}` : 'Memory telemetry unavailable' },
    { label: 'GPU', value: gpu ? `${gpu.utilization_percent}%` : '—', percent: gpu?.utilization_percent ?? null, title: gpu ? `${gpu.name} compute utilization · sampled ${value?.sampled_at}` : 'No NVIDIA telemetry detected' },
    { label: 'VRAM', value: gpu ? `${gpu.percent}%` : '—', percent: gpu?.percent ?? null, title: gpu ? `${gpu.name} memory · ${bytesLabel(gpu.used_bytes)} of ${bytesLabel(gpu.total_bytes)} · sampled ${value?.sampled_at}` : 'No NVIDIA telemetry detected' },
    { label: 'CTX', value: value?.context?.remaining_tokens == null ? '—' : `${Math.round(value.context.remaining_tokens / 1000)}k free`, percent: value?.context?.percent ?? null, title: value?.context?.limit_tokens ? `${value.context.used_tokens ?? 0} of ${value.context.limit_tokens} tokens used · ${value.context.source}` : value?.context?.source ?? 'Context usage unavailable', always: true },
  ].filter(item => item.always || (item.percent != null && item.percent > 0))
  return <div className="resource-strip" aria-label="Local resource usage">{items.map(item => <span key={item.label} title={item.title}><i>{item.label}</i><b>{item.value}</b><em><u style={{ width: `${item.percent ?? 0}%` }} /></em></span>)}</div>
}

function AttentionPanel({
  attention,
  busy,
  onResolve,
}: {
  attention: Attention
  busy: boolean
  onResolve: (key: string, text?: string) => void
}) {
  const [custom, setCustom] = useState('')
  return (
    <motion.section className={`attention attention-${attention.kind}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }}>
      <div className="attention-copy">
        <span className="eyebrow">{attention.details || statusLabel(attention.kind)}</span>
        <h2>{attention.title}</h2>
        {attention.message && <p>{attention.message}</p>}
        {attention.kind === 'question' && <small className="attention-scope">Access controls tool approvals; this choice defines the product.</small>}
      </div>
      <div className="attention-options">
        {attention.options.map(option => (
          <button key={option.key} className={option.primary ? 'primary' : ''} disabled={busy} onClick={() => onResolve(option.key)}>
            <span>{option.label}</span>{option.description && <small>{option.description}</small>}
          </button>
        ))}
      </div>
      {attention.allow_custom && (
        <form className="custom-answer" onSubmit={event => { event.preventDefault(); if (custom.trim() && !busy) onResolve('custom', custom) }}>
          <input value={custom} onChange={event => setCustom(event.target.value)} placeholder="Write another answer" />
          <IconButton type="submit" label="Submit custom answer" disabled={!custom.trim() || busy}><Send size={15} /></IconButton>
        </form>
      )}
    </motion.section>
  )
}

function Composer({
  disabled,
  running,
  activeJobId,
  value,
  queueCount,
  sendCapability,
  modeCapability,
  pauseCapability,
  killCapability,
  onSubmit,
  onGuide,
  onValueChange,
  onStop,
  onKill,
  showQuickStarts,
  workflowMode,
  access,
  onModeChange,
  onAccessChange,
}: {
  disabled: boolean
  running: boolean
  activeJobId?: string
  value: string
  queueCount: number
  sendCapability?: ActionCapability
  modeCapability?: ActionCapability
  pauseCapability?: ActionCapability
  killCapability?: ActionCapability
  onSubmit: (text: string) => void
  onGuide: (text: string) => void
  onValueChange: (text: string) => void
  onStop: (jobId: string) => void
  onKill: () => void
  showQuickStarts: boolean
  workflowMode: 'plan' | 'normal' | 'ultra'
  access: 'normal' | 'bounded' | 'full' | 'host'
  onModeChange: (mode: 'plan' | 'normal' | 'ultra') => void
  onAccessChange: (access: 'default' | 'bounded' | 'full') => void
}) {
  const [accessOpen, setAccessOpen] = useState(false)
  const [modeOpen, setModeOpen] = useState(false)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const pickerRoot = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent && event.key !== 'Escape') return
      if (event instanceof MouseEvent && pickerRoot.current?.contains(event.target as Node)) return
      setAccessOpen(false); setModeOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', close) }
  }, [])
  const matches = value.startsWith('/')
    ? commands.filter(([command]) => command.startsWith(value.toLowerCase()) || value === '/')
    : []
  const submit = () => {
    const text = value.trim()
    if (!text || disabled || sendCapability?.allowed === false) return
    onSubmit(text)
    requestAnimationFrame(() => textarea.current?.focus())
  }
  return (
    <div className="composer-wrap" ref={pickerRoot}>
      <AnimatePresence>
        {matches.length > 0 && (
          <motion.div className="command-menu" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
            {matches.slice(0, 7).map(([command, description]) => (
              <button key={command} onClick={() => { onValueChange(command); textarea.current?.focus() }}>
                <code>{command.trim()}</code><span>{description}</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
      <div className={`composer ${running ? 'is-running' : ''}`}>
        <textarea
          ref={textarea}
          dir="auto"
          rows={1}
          value={value}
          disabled={disabled}
          onChange={event => onValueChange(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault(); submit()
            }
          }}
          placeholder={running ? `Add the next task to queue${queueCount ? ` · ${queueCount}/10 waiting` : ''}` : workflowMode === 'plan' ? 'Describe the plan or tell GA3BAD what to revise' : workflowMode === 'ultra' ? 'Give Ultra a complex goal with clear acceptance criteria' : 'Ask GA3BAD to build, fix, explain, or review'}
          aria-label="Task message"
        />
        <div className="composer-toolbar">
          <div className="composer-context">
            <div className="access-picker">
              <button className="composer-control permission-control" aria-haspopup="menu" aria-expanded={accessOpen} onClick={() => { setAccessOpen(open => !open); setModeOpen(false) }}><Shield size={15} /><span>{access === 'full' ? 'Docker Full' : access === 'host' ? 'Host Full' : access === 'bounded' ? 'Bounded' : 'Default'}</span><ChevronDown size={12} /></button>
              <AnimatePresence>
                {accessOpen && <motion.div className="access-menu" role="menu" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}>
                  {([
                    ['full', 'Full Access', 'Choose Docker Full or task-scoped Host Full'],
                    ['default', 'Default', 'Ask according to action risk'],
                    ['bounded', 'Bounded', 'Approval-gated workspace access'],
                  ] as const).map(([key, label, description]) => (
                    <button key={key} role="menuitem" className={(key === 'default' ? access === 'normal' : key === 'full' ? access === 'full' || access === 'host' : access === key) ? 'selected' : ''} onClick={() => { onAccessChange(key); setAccessOpen(false) }}><span>{(key === 'default' ? access === 'normal' : key === 'full' ? access === 'full' || access === 'host' : access === key) && <Check size={13} />}{label}</span><small>{description}</small></button>
                  ))}
                </motion.div>}
              </AnimatePresence>
            </div>
            <div className="mode-picker">
              <button className={`composer-control mode-control mode-${workflowMode}`} aria-haspopup="menu" aria-expanded={modeOpen} disabled={modeCapability?.allowed === false} title={modeCapability?.allowed === false ? `${modeCapability.reason} ${modeCapability.remediation}` : 'Choose workflow mode'} onClick={() => { setModeOpen(open => !open); setAccessOpen(false) }}><Sparkles size={14} /><span>Mode · {workflowMode}</span><ChevronDown size={12} /></button>
              <AnimatePresence>{modeOpen && <motion.div className="access-menu mode-menu" role="menu" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}>
                {([
                  ['plan', 'Plan', 'Create and revise without executing'],
                  ['normal', 'Normal', 'One durable goal with review and evidence'],
                  ['ultra', 'Ultra', 'Recursive specialists and deeper quality gates'],
                ] as const).map(([key, label, description]) => <button key={key} role="menuitem" className={workflowMode === key ? 'selected' : ''} onClick={() => { onModeChange(key); setModeOpen(false) }}><span>{workflowMode === key && <Check size={13} />}{label}</span><small>{description}</small></button>)}
              </motion.div>}</AnimatePresence>
            </div>
          </div>
          <div className="composer-actions">
            {running && value.trim() && <button className="guide-button" onClick={() => onGuide(value.trim())} title="Apply to the running task at its next safe checkpoint"><MessageSquarePlus size={14} />Guide current</button>}
            {running && activeJobId && <button className="pause-button" disabled={pauseCapability?.allowed === false} onClick={() => onStop(activeJobId)} aria-label={pauseCapability?.allowed === false ? 'Pause requested' : 'Pause safely'} title={pauseCapability?.allowed === false ? `${pauseCapability.reason} ${pauseCapability.remediation}` : 'Pause after the current action reaches a safe checkpoint'}>{pauseCapability?.allowed === false ? <LoaderCircle className="spin" size={15} /> : <Pause size={15} />}</button>}
            {killCapability?.allowed && <button className="kill-button" onClick={onKill} aria-label="Kill task immediately" title="Kill immediately without waiting for a safe checkpoint"><OctagonX size={15} /></button>}
            <button className="send-button" disabled={!value.trim() || disabled || sendCapability?.allowed === false} onClick={submit} aria-label={running ? 'Add to queue' : 'Send message'} title={sendCapability?.allowed === false ? `${sendCapability.reason} ${sendCapability.remediation}` : running ? 'Add to queue' : 'Send'}>{running ? <ListPlus size={16} /> : <Send size={16} />}</button>
          </div>
        </div>
      </div>
      <p className="composer-hint">{sendCapability?.allowed === false ? `${sendCapability.reason} ${sendCapability.remediation}` : running ? 'Enter adds to queue · Guide current applies at the next safe checkpoint' : 'Enter to send · Shift+Enter for a new line · / for commands'}</p>
      {showQuickStarts && (
        <div className="quick-starts">
          <button onClick={() => onSubmit('Explain this project to me')}><Sparkles size={14} />Explain this project to me</button>
          <button onClick={() => onSubmit('Analyze the workspace and suggest the safest first change')}><Shield size={14} />Analyze the workspace and suggest the safest first change</button>
        </div>
      )}
    </div>
  )
}

function WorkspaceInspector({ threadId, name, onClose }: { threadId: string; name: string; onClose: () => void }) {
  const [selectedFile, setSelectedFile] = useState('')
  const query = useQuery({
    queryKey: ['inspector', threadId, name],
    queryFn: () => api.inspector(threadId, name),
  })
  const previewQuery = useQuery({
    queryKey: ['file-preview', threadId, selectedFile],
    queryFn: () => api.filePreview(threadId, selectedFile),
    enabled: name === 'files' && Boolean(selectedFile),
  })
  const content = query.data ?? {}
  const files = Array.isArray(content.files) ? content.files as Array<{ path: string; size: number }> : []
  const raw = name === 'changes' ? String(content.diff ?? 'No changes.') : JSON.stringify(content, null, 2)
  return (
    <motion.aside className="inspector" initial={{ x: 24, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 24, opacity: 0 }}>
      <div className="inspector-heading"><div><span className="eyebrow">Inspector</span><h2>{inspectors.find(item => item[0] === name)?.[1] ?? name}</h2></div><IconButton label="Close inspector" onClick={onClose}><X size={17} /></IconButton></div>
      {query.isError ? <div className="inspector-loading error">Unable to load this inspector: {String(query.error)}</div> : query.isLoading ? <div className="inspector-loading">Loading durable state…</div> : name === 'files' ? (
        <div className="file-explorer">
          <div className="file-root"><FolderOpen size={14} /><span>{String(content.workspace ?? 'Workspace')}</span></div>
          {selectedFile ? <div className="file-preview"><div><button aria-label="Back to workspace files" onClick={() => setSelectedFile('')}><ChevronRight size={14} /></button><strong>{selectedFile}</strong></div>{previewQuery.isLoading ? <p>Loading safe text preview…</p> : previewQuery.isError ? <p className="error">{String(previewQuery.error)}</p> : <pre>{previewQuery.data?.content}</pre>}</div> : files.map(file => <button key={file.path} aria-label={`Preview ${file.path}`} onClick={() => setSelectedFile(file.path)}><span>{file.path}</span><small>{file.size < 1024 ? `${file.size} B` : `${Math.ceil(file.size / 1024)} KB`}</small></button>)}
          {!files.length && <p>No workspace files found.</p>}
        </div>
      ) : <pre className={name === 'changes' ? 'diff-output' : ''}>{raw}</pre>}
    </motion.aside>
  )
}

function TerminalPanel({ threadId, onClose }: { threadId: string; onClose: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState('opening')
  useEffect(() => {
    let disposed = false
    let terminal: import('@xterm/xterm').Terminal | null = null
    let subscription: { dispose: () => void } | null = null
    let fit: import('@xterm/addon-fit').FitAddon | null = null
    let session: TerminalSession | null = null
    let line = ''
    const start = async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([import('@xterm/xterm'), import('@xterm/addon-fit')])
      if (disposed || !host.current) return
      terminal = new Terminal({ convertEol: true, cursorBlink: true, fontSize: 12, fontFamily: 'Cascadia Mono, Consolas, monospace', theme: { background: '#111313', foreground: '#c7cec9', cursor: '#f06a2a' } })
      fit = new FitAddon(); terminal.loadAddon(fit); terminal.open(host.current); fit.fit()
      terminal.writeln('GA3BAD restricted workspace terminal')
      terminal.writeln('Default/Bounded allow inspection and verification commands.\r\n')
      try {
        const value = await api.openTerminal(threadId)
        if (disposed || !terminal) return
        session = value; setMode(value.mode)
        if (value.scrollback) terminal.write(value.scrollback)
        terminal.write(`\r\n${value.mode}:${value.cwd}> `)
      } catch (error) {
        terminal?.writeln(`\r\nUnable to open terminal: ${String(error)}`)
      }
      subscription = terminal?.onData(data => {
        if (!session || !terminal) return
        if (data === '\r') {
          const command = line.trim(); line = ''; terminal.write('\r\n')
          if (!command) { terminal.write(`${session.mode}:${session.cwd}> `); return }
          terminal.write('running…\r\n')
          void api.terminalCommand(session.id, command).then(result => {
            session = result
            terminal?.write(result.output || '(no output)'); terminal?.write(`\r\n[exit ${result.returncode}]\r\n${result.mode}:${result.cwd}> `)
          }).catch(error => terminal?.write(`Blocked: ${String(error)}\r\n${session!.mode}:${session!.cwd}> `))
        } else if (data === '\u007F') {
          if (line) { line = line.slice(0, -1); terminal.write('\b \b') }
        } else if (data >= ' ') { line += data; terminal.write(data) }
      }) ?? null
    }
    void start()
    const resize = () => fit?.fit(); window.addEventListener('resize', resize)
    return () => { disposed = true; subscription?.dispose(); window.removeEventListener('resize', resize); terminal?.dispose() }
  }, [threadId])
  return (
    <motion.aside className="inspector terminal-panel" initial={{ x: 24, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 24, opacity: 0 }}>
      <div className="inspector-heading"><div><span className="eyebrow">{mode} boundary</span><h2>Terminal</h2></div><IconButton label="Close terminal" onClick={onClose}><X size={17} /></IconButton></div>
      <div className="xterm-host" ref={host} aria-label="Project terminal" />
    </motion.aside>
  )
}

const fallbackInferenceProfile: InferenceProfile = {
  device: 'auto', context_window: 16_384, max_output_tokens: 4_096,
  gpu_layers: -1, cpu_threads: 4, temperature: 0.2, top_p: 0.9,
  top_k: 40, performance: 'balanced', estimated_minutes_per_step: 30,
  planning_steps: 16, work_quantum_steps: 24, review_steps: 12,
  max_provider_retries: 3, ultra_cloud_concurrency: 4, ultra_max_depth: 8,
}

function SettingsDialog({ settings, threadId, onClose, onSave }: { settings: Settings; threadId: string | null; onClose: () => void; onSave: (patch: Partial<Settings>) => void }) {
  const [tab, setTab] = useState<'interface' | 'models' | 'execution' | 'advanced'>('interface')
  const [scope, setScope] = useState<'default' | 'task'>(threadId ? 'task' : 'default')
  const modelQuery = useQuery({ queryKey: ['model-settings', threadId], queryFn: () => api.modelSettings(threadId ?? undefined), enabled: tab === 'models' })
  const dockerQuery = useQuery({ queryKey: ['docker-status'], queryFn: api.dockerStatus, enabled: tab === 'execution' })
  const advancedQuery = useQuery({ queryKey: ['advanced-settings'], queryFn: api.advancedSettings, enabled: tab === 'advanced' })
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [inference, setInference] = useState<InferenceProfile>(settings.inference_profile ?? fallbackInferenceProfile)
  const [savingRole, setSavingRole] = useState('')
  const [modelNote, setModelNote] = useState('')
  const [advancedNote, setAdvancedNote] = useState('')
  useEffect(() => {
    const data = modelQuery.data
    if (!data) return
    const source = scope === 'task' ? { ...data.defaults, ...data.overrides } : data.defaults
    setDrafts({ main: source.main ?? '', router: source.router ?? '', verifier: source.verifier ?? '', embedding: source.embedding ?? data.embedding_default ?? '' })
  }, [modelQuery.data, scope])
  useEffect(() => { if (advancedQuery.data) setInference(advancedQuery.data) }, [advancedQuery.data])
  const saveRole = async (role: string) => {
    setSavingRole(role); setModelNote('')
    try {
      if (scope === 'task' && threadId) await api.setThreadModelRole(threadId, role, drafts[role] ?? '')
      else await api.setDefaultModelRole(role, drafts[role] ?? '')
      setModelNote(`${role} saved at a safe checkpoint.`)
      await modelQuery.refetch()
    } catch (error) { setModelNote(String(error)) } finally { setSavingRole('') }
  }
  const setupDocker = async () => {
    await api.setupDocker(); await dockerQuery.refetch()
  }
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <motion.div className="settings-dialog" role="dialog" aria-modal="true" aria-label="Settings" onMouseDown={event => event.stopPropagation()} initial={{ opacity: 0, scale: .98 }} animate={{ opacity: 1, scale: 1 }}>
        <div className="dialog-heading"><div><span className="eyebrow">Local application</span><h2>Settings</h2></div><IconButton label="Close settings" onClick={onClose}><X size={18} /></IconButton></div>
        <nav className="settings-tabs" aria-label="Settings sections">{([['interface', Monitor, 'Interface'], ['models', Bot, 'Models'], ['execution', Container, 'Execution'], ['advanced', Gauge, 'Advanced']] as const).map(([value, Icon, label]) => <button className={tab === value ? 'active' : ''} key={value} onClick={() => setTab(value)}><Icon size={15} />{label}</button>)}</nav>
        {tab === 'interface' && <section className="settings-page"><h3>Interface</h3><p>Application language is separate from the direction of individual model outputs.</p><div className="segmented">
          {([['dark', Moon], ['light', Sun], ['system', Monitor]] as const).map(([value, Icon]) => <button key={value} className={settings.theme === value ? 'active' : ''} onClick={() => onSave({ theme: value })}><Icon size={15} />{value}</button>)}
        </div><div className="settings-row language-row"><label>Application language</label><select aria-label="Application language" value={settings.locale} onChange={event => onSave({ locale: event.target.value as Settings['locale'] })}><option value="auto">Auto</option><option value="en">English</option><option value="ar">العربية</option></select></div><div className="settings-row"><label>Technical details</label><select value={settings.experience} onChange={event => onSave({ experience: event.target.value as Settings['experience'] })}><option value="simple">Simple</option><option value="advanced">Advanced</option></select></div><label className="toggle-row"><span>Reduced motion</span><input type="checkbox" checked={settings.reduced_motion} onChange={event => onSave({ reduced_motion: event.target.checked })} /></label></section>}
        {tab === 'models' && <section className="settings-page model-settings"><div className="settings-page-heading"><div><h3>Model roles</h3><p>Main is required. Router and Verifier remain isolated roles; Embedding falls back to deterministic hashing.</p></div><button onClick={async () => { await api.refreshModels(); await modelQuery.refetch() }}><RefreshCw size={14} />Refresh</button></div><div className="segmented model-scope"><button className={scope === 'default' ? 'active' : ''} onClick={() => setScope('default')}>New task default</button><button disabled={!threadId} className={scope === 'task' ? 'active' : ''} onClick={() => setScope('task')}>This task</button></div>
          {(['main', 'router', 'verifier', 'embedding'] as const).map(role => <div className="model-role" key={role}><div><strong>{role}</strong><small>{role === 'main' ? 'Planning and implementation' : role === 'router' ? 'Fast typed intake enrichment' : role === 'verifier' ? 'Independent review context' : 'Repository semantic index'}</small></div>{role === 'embedding' ? <input value={drafts[role] ?? ''} onChange={event => setDrafts(value => ({ ...value, [role]: event.target.value }))} placeholder="nomic-embed-text:latest" /> : <select value={drafts[role] ?? ''} onChange={event => setDrafts(value => ({ ...value, [role]: event.target.value }))}><option value="">{role === 'main' ? 'Use environment model' : 'Use Main / deterministic fallback'}</option>{modelQuery.data?.models.map(model => <option key={model.id} value={model.id}>{model.model} · {model.provider} · {model.execution_class}</option>)}</select>}<button disabled={savingRole === role || (role === 'main' && !drafts.main)} onClick={() => void saveRole(role)}>{savingRole === role ? 'Saving…' : 'Save'}</button></div>)}
          {modelQuery.data?.diagnostics.map(item => <p className="settings-diagnostic" key={`${item.source}-${item.message}`}>{item.source}: {item.message}</p>)}{modelNote && <p className="settings-note">{modelNote}</p>}</section>}
        {tab === 'execution' && <section className="settings-page execution-settings"><div className="execution-status"><Container size={20} /><div><h3>Docker container</h3><p>{dockerQuery.data?.ready ? 'Ready for isolated Full Access.' : String(dockerQuery.data?.reason ?? 'Checking the local Docker sandbox…')}</p></div><span className={dockerQuery.data?.ready ? 'ready' : ''}>{dockerQuery.data?.ready ? 'Ready' : 'Not ready'}</span></div><dl><div><dt>Image</dt><dd>{String(dockerQuery.data?.image ?? 'ga3bad/coding-agent-sandbox')}</dd></div><div><dt>Docker</dt><dd>{String(dockerQuery.data?.docker_version ?? 'Unavailable')}</dd></div></dl><button className="setup-docker" onClick={() => void setupDocker()}><Container size={15} />Set up Docker sandbox</button><p>Docker setup creates the versioned non-root GA3BAD image. No VirtualBox integration is used.</p></section>}
        {tab === 'advanced' && <section className="settings-page advanced-settings"><div className="settings-page-heading"><div><h3>Advanced inference</h3><p>Controls supported local Ollama runners. Cloud providers may ignore hardware-specific values.</p></div><Gauge size={20} /></div><div className="advanced-grid">
          <label><span>Compute device</span><select value={inference.device} onChange={event => setInference(value => ({ ...value, device: event.target.value as InferenceProfile['device'] }))}><option value="auto">Auto</option><option value="gpu">GPU</option><option value="cpu">CPU</option></select></label>
          <label><span>Performance</span><select value={inference.performance} onChange={event => setInference(value => ({ ...value, performance: event.target.value as InferenceProfile['performance'] }))}><option value="eco">Eco</option><option value="balanced">Balanced</option><option value="performance">Performance</option></select></label>
          <label><span>Context window</span><input type="number" min="2048" max="131072" step="1024" value={inference.context_window} onChange={event => setInference(value => ({ ...value, context_window: Number(event.target.value) }))} /></label>
          <label><span>Max output tokens</span><input type="number" min="128" max="65536" step="128" value={inference.max_output_tokens} onChange={event => setInference(value => ({ ...value, max_output_tokens: Number(event.target.value) }))} /></label>
          <label><span>GPU layers <small>−1 = automatic/all</small></span><input type="number" min="-1" max="999" value={inference.gpu_layers} onChange={event => setInference(value => ({ ...value, gpu_layers: Number(event.target.value) }))} /></label>
          <label><span>CPU threads</span><input type="number" min="1" max="256" value={inference.cpu_threads} onChange={event => setInference(value => ({ ...value, cpu_threads: Number(event.target.value) }))} /></label>
          <label><span>Temperature</span><input type="number" min="0" max="2" step="0.05" value={inference.temperature} onChange={event => setInference(value => ({ ...value, temperature: Number(event.target.value) }))} /></label>
          <label><span>Top P</span><input type="number" min="0" max="1" step="0.05" value={inference.top_p} onChange={event => setInference(value => ({ ...value, top_p: Number(event.target.value) }))} /></label>
          <label><span>Top K</span><input type="number" min="0" max="1000" value={inference.top_k} onChange={event => setInference(value => ({ ...value, top_k: Number(event.target.value) }))} /></label>
          <label><span>Estimate per plan step <small>minutes</small></span><input type="number" min="1" max="720" value={inference.estimated_minutes_per_step} onChange={event => setInference(value => ({ ...value, estimated_minutes_per_step: Number(event.target.value) }))} /></label>
          <h4>Harness behavior</h4>
          <label><span>Planning step budget</span><input type="number" min="2" max="100" value={inference.planning_steps} onChange={event => setInference(value => ({ ...value, planning_steps: Number(event.target.value) }))} /></label>
          <label><span>Normal work quantum</span><input type="number" min="1" max="500" value={inference.work_quantum_steps} onChange={event => setInference(value => ({ ...value, work_quantum_steps: Number(event.target.value) }))} /></label>
          <label><span>Review step budget</span><input type="number" min="2" max="100" value={inference.review_steps} onChange={event => setInference(value => ({ ...value, review_steps: Number(event.target.value) }))} /></label>
          <label><span>Provider retries</span><input type="number" min="0" max="10" value={inference.max_provider_retries} onChange={event => setInference(value => ({ ...value, max_provider_retries: Number(event.target.value) }))} /></label>
          <label><span>Ultra cloud concurrency</span><input type="number" min="1" max="8" value={inference.ultra_cloud_concurrency} onChange={event => setInference(value => ({ ...value, ultra_cloud_concurrency: Number(event.target.value) }))} /></label>
          <label><span>Ultra recursion depth</span><input type="number" min="1" max="12" value={inference.ultra_max_depth} onChange={event => setInference(value => ({ ...value, ultra_max_depth: Number(event.target.value) }))} /></label>
        </div><div className="advanced-warning"><Cpu size={17} /><span>Changes reload idle model sessions. Active work must reach a safe checkpoint first.</span></div>{advancedNote && <p className="settings-note">{advancedNote}</p>}<button className="save-advanced" onClick={async () => { setAdvancedNote(''); try { await api.patchAdvancedSettings(inference); await advancedQuery.refetch(); setAdvancedNote('Saved. Idle local model sessions will use this profile.') } catch (error) { setAdvancedNote(String(error)) } }}><Check size={15} />Save advanced settings</button></section>}
      </motion.div>
    </div>
  )
}

function FullAccessDialog({ onClose, onApply }: { onClose: () => void; onApply: (policy: 'full' | 'host', token?: string) => Promise<Record<string, unknown>> }) {
  const [policy, setPolicy] = useState<'full' | 'host' | null>(null)
  const [challenge, setChallenge] = useState<Record<string, unknown> | null>(null)
  const [busy, setBusy] = useState(false)
  const [acknowledged, setAcknowledged] = useState(false)
  const [error, setError] = useState('')
  const prepare = async (value: 'full' | 'host') => { setPolicy(value); setBusy(true); setError(''); try { setChallenge(await onApply(value)) } catch (reason) { setError(String(reason)) } finally { setBusy(false) } }
  const confirm = async () => { if (!policy || !challenge) return; setBusy(true); setError(''); try { await onApply(policy, String(challenge.confirmation_token ?? '')); onClose() } catch (reason) { setError(String(reason)); setBusy(false) } }
  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}><motion.div className="full-access-dialog" role="dialog" aria-modal="true" aria-label="Full Access warning" onMouseDown={event => event.stopPropagation()} initial={{ opacity: 0, scale: .98 }} animate={{ opacity: 1, scale: 1 }}><div className="dialog-heading"><div><span className="eyebrow">Permission boundary</span><h2>Choose Full Access</h2></div><IconButton label="Close" onClick={onClose}><X size={18} /></IconButton></div><p>Full Access is independent from Mode. Plan remains read-only until you implement the approved revision.</p><div className="full-access-options"><button className={policy === 'full' ? 'selected' : ''} onClick={() => void prepare('full')}><Container size={18} /><strong>Docker Full</strong><small>Skip repeated approvals inside the ready isolated container.</small></button><button className={policy === 'host' ? 'selected danger' : ''} onClick={() => void prepare('host')}><Server size={18} /><strong>Host Full</strong><small>Run directly with your Windows user permissions for this task only.</small></button></div>{challenge && <div className="access-warning"><Shield size={18} /><p>{String(challenge.warning ?? '')}</p></div>} {policy === 'host' && challenge && <label className="acknowledge"><input type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />I understand that host commands are not container-isolated.</label>}{error && <div className="form-error">{error}</div>}<div className="dialog-actions"><button onClick={onClose}>Cancel</button><button className="primary" disabled={!challenge || busy || (policy === 'host' && !acknowledged)} onClick={() => void confirm()}>{busy ? 'Checking…' : `Enable ${policy === 'host' ? 'Host' : 'Docker'} Full`}</button></div></motion.div></div>
}

function KillTaskDialog({ title, busy, onClose, onConfirm }: { title: string; busy: boolean; onClose: () => void; onConfirm: () => void }) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}><motion.div className="kill-task-dialog" role="alertdialog" aria-modal="true" aria-labelledby="kill-task-title" onMouseDown={event => event.stopPropagation()} initial={{ opacity: 0, scale: .98, y: 5 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: .98, y: 4 }}><div className="kill-task-mark"><OctagonX size={21} /></div><div><span className="eyebrow">Immediate termination</span><h2 id="kill-task-title">Kill “{title}”?</h2><p>This does not wait for a safe checkpoint. Active model generation and managed processes will be terminated, queued prompts will be cancelled, and uncertain file effects may remain for review.</p></div><div className="dialog-actions"><button disabled={busy} onClick={onClose}>Keep task</button><button className="danger" disabled={busy} onClick={onConfirm}>{busy ? 'Killing…' : 'Kill task now'}</button></div></motion.div></div>
}

function ProjectDialog({ onClose, onSubmit, busy, error }: { onClose: () => void; onSubmit: (path: string) => void; busy: boolean; error?: string }) {
  const [path, setPath] = useState('')
  const [picking, setPicking] = useState(false)
  const [pickerError, setPickerError] = useState('')
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <motion.form className="project-dialog" role="dialog" aria-modal="true" aria-label="Add project" onSubmit={event => { event.preventDefault(); onSubmit(path) }} onMouseDown={event => event.stopPropagation()} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <div className="dialog-heading"><div><span className="eyebrow">Local folder</span><h2>Add project</h2></div><IconButton label="Close" type="button" onClick={onClose}><X size={18} /></IconButton></div>
        <p>Enter the absolute path to an existing project folder on this computer.</p>
        <div className="project-path-picker"><input autoFocus value={path} onChange={event => setPath(event.target.value)} placeholder="D:\projects\my-app" /><button type="button" disabled={picking} onClick={async () => { setPicking(true); setPickerError(''); try { const result = await api.pickFolder(); if (result.path) setPath(result.path) } catch (pickError) { setPickerError(String(pickError)) } finally { setPicking(false) } }}><FolderOpen size={16} />{picking ? 'Opening…' : 'Browse'}</button></div>
        {(error || pickerError) && <div className="form-error">{error || pickerError}</div>}
        <div className="dialog-actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary" disabled={!path.trim() || busy}>{busy ? 'Checking…' : 'Add project'}</button></div>
      </motion.form>
    </div>
  )
}

export default function App() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const selectedThreadId = useSelectedThreadId()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [online, setOnline] = useState(false)
  const [inspector, setInspector] = useState<string | null>(null)
  const [terminalOpen, setTerminalOpen] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showProject, setShowProject] = useState(false)
  const [showFullAccess, setShowFullAccess] = useState(false)
  const [showKillTask, setShowKillTask] = useState(false)
  const [draftText, setDraftText] = useState('')
  const draftThread = useRef<string | null>(null)
  const draftRevision = useRef(0)
  const transcriptEnd = useRef<HTMLDivElement>(null)

  const bootstrapQuery = useQuery({ queryKey: ['bootstrap'], queryFn: getBootstrap })
  const threadQuery = useQuery({
    queryKey: ['thread', selectedThreadId],
    queryFn: () => api.getThread(selectedThreadId!),
    enabled: Boolean(selectedThreadId),
    refetchInterval: 15_000,
  })
  const bootstrap = bootstrapQuery.data
  const snapshot = threadQuery.data
  const changesQuery = useQuery({
    queryKey: ['inspector', selectedThreadId, 'changes'],
    queryFn: () => api.inspector(selectedThreadId!, 'changes'),
    enabled: Boolean(selectedThreadId && snapshot?.dashboard?.tasks?.length),
  })
  const visualizationQuery = useQuery({
    queryKey: ['visualization', selectedThreadId],
    queryFn: () => api.visualization(selectedThreadId!),
    enabled: Boolean(selectedThreadId && snapshot?.thread.view_mode === 'visualize'),
    refetchInterval: snapshot?.thread.view_mode === 'visualize' ? 2_000 : false,
  })

  useEffect(() => connectEvents(event => {
    if (event.type !== 'app.ping') void queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    if (event.thread_id) void queryClient.invalidateQueries({ queryKey: ['thread', event.thread_id] })
    if (event.thread_id && inspector) void queryClient.invalidateQueries({ queryKey: ['inspector', event.thread_id, inspector] })
  }, setOnline, () => {
    void queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    if (selectedThreadId) void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
  }), [queryClient, inspector, selectedThreadId])

  useEffect(() => {
    if (!bootstrap || selectedThreadId) return
    const first = bootstrap.projects.flatMap(project => project.threads ?? []).find(thread => !thread.archived)
    if (first) navigate(taskPath(first.id), { replace: true })
  }, [bootstrap, selectedThreadId, navigate])

  useEffect(() => {
    const target = transcriptEnd.current
    if (target && typeof target.scrollIntoView === 'function') {
      target.scrollIntoView({ block: 'end', behavior: 'smooth' })
    }
  }, [snapshot?.messages.length, snapshot?.presentation?.activity.summary])

  useEffect(() => {
    if (!selectedThreadId || !snapshot?.draft || draftThread.current === selectedThreadId) return
    draftThread.current = selectedThreadId
    draftRevision.current = snapshot.draft.revision
    setDraftText(snapshot.draft.text ?? '')
  }, [selectedThreadId, snapshot?.draft])

  useEffect(() => {
    if (!selectedThreadId || draftThread.current !== selectedThreadId) return
    const timer = window.setTimeout(() => {
      void api.saveDraft(selectedThreadId, draftText, draftRevision.current).then(value => { draftRevision.current = value.revision }).catch(() => {})
    }, 450)
    return () => window.clearTimeout(timer)
  }, [selectedThreadId, draftText])

  const addProject = useMutation({
    mutationFn: (path: string) => api.addProject(path),
    onSuccess: () => { setShowProject(false); void queryClient.invalidateQueries({ queryKey: ['bootstrap'] }) },
  })
  const createThread = useMutation({
    mutationFn: (projectId: string) => api.createThread(projectId),
    onSuccess: (thread: unknown) => {
      const value = thread as ThreadSummary
      void queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
      navigate(taskPath(value.id))
      setSidebarOpen(false)
    },
  })
  const submitInput = useMutation({
    mutationFn: ({ text, delivery = 'queue' }: { text: string; delivery?: 'queue' | 'guidance' }) => api.submit(selectedThreadId!, text, delivery),
    onSuccess: () => {
      setDraftText('')
      void queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
      void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
    },
  })
  const cancelQueued = useMutation({
    mutationFn: (jobId: string) => api.cancelQueued(selectedThreadId!, jobId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })
  const continueQueue = useMutation({
    mutationFn: () => api.continueQueue(selectedThreadId!),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })
  const resumeThread = useMutation({
    mutationFn: () => api.resume(selectedThreadId!),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })
  const killTask = useMutation({
    mutationFn: () => api.killTask(selectedThreadId!),
    onSuccess: async () => {
      setShowKillTask(false)
      await queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
      await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    },
  })
  const viewMutation = useMutation({
    mutationFn: (view: 'transcript' | 'visualize') => api.setView(selectedThreadId!, view, snapshot!.thread.state_revision),
    onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }); void queryClient.invalidateQueries({ queryKey: ['bootstrap'] }) },
  })
  const resolveAttention = useMutation({
    mutationFn: ({ key, text = '' }: { key: string; text?: string }) => api.resolveAttention(selectedThreadId!, snapshot!.presentation!.attention!.id, key, text),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })
  const modeMutation = useMutation({
    mutationFn: (mode: 'plan' | 'normal' | 'ultra') => api.setMode(selectedThreadId!, mode, snapshot!.thread.state_revision),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
      await queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
    },
  })
  const directionMutation = useMutation({
    mutationFn: ({ message, direction }: { message: Message; direction: 'auto' | 'rtl' }) => api.setMessageDirection(selectedThreadId!, message.id, direction),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })
  const planDecision = useMutation({
    mutationFn: (body: { action: 'implement' | 'keep_planning'; feedback?: string }) => api.planDecision(selectedThreadId!, {
      ...body,
      revision: snapshot!.dashboard!.plan_revision,
      fingerprint: String(snapshot!.dashboard!.plan_fingerprint ?? ''),
    }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] }),
  })

  const settings = bootstrap?.settings ?? ({ theme: 'dark', locale: 'auto', experience: 'simple', mode: 'normal', access: 'normal', reduced_motion: false } satisfies Settings)
  useEffect(() => {
    const resolved = settings.theme === 'system'
      ? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
      : settings.theme
    const browserLanguage = navigator.language.toLowerCase().split('-')[0]
    const locale = settings.locale === 'auto' ? (browserLanguage === 'ar' ? 'ar' : 'en') : settings.locale
    document.documentElement.dataset.theme = resolved
    // Interface layout remains stable. RTL is an explicit per-assistant-message preference.
    document.documentElement.dir = 'ltr'
    document.documentElement.lang = locale
    document.documentElement.dataset.reducedMotion = settings.reduced_motion ? 'true' : 'false'
  }, [settings])

  const selectedProject = useMemo(() => bootstrap?.projects.find(project => project.id === snapshot?.thread.project_id), [bootstrap, snapshot])
  const activeJob = snapshot?.jobs.find(job => job.status === 'running') ?? snapshot?.jobs[0]
  const running = activeJob?.status === 'running' || Boolean(snapshot?.presentation?.running)
  const waitingForAnswer = snapshot?.presentation?.attention?.kind === 'question'
  const workState = waitingForAnswer
    ? 'Waiting for your answer'
    : running
      ? (snapshot?.presentation?.activity.summary || 'Working')
      : snapshot?.dashboard?.status === 'awaiting_plan_approval'
        ? 'Plan ready for review'
        : snapshot?.thread.status === 'paused'
          ? 'Paused safely'
          : snapshot?.thread.status === 'killed'
            ? 'Killed'
          : snapshot?.thread.status === 'recovery_required'
            ? 'Recovery required'
            : snapshot?.thread.status === 'problem'
              ? 'Needs attention'
      : snapshot?.thread.status === 'completed'
        ? 'Completed'
        : 'Ready'
  const workflowMode = snapshot?.thread.workflow_mode ?? settings.mode
  const effectiveAccess = snapshot?.thread.effective_access ?? (snapshot?.thread.access_policy === 'full' ? 'full' : snapshot?.thread.access_policy === 'bounded' ? 'bounded' : 'normal')
  const turns = useMemo(() => {
    if (snapshot?.turns?.length) return snapshot.turns.map(turn => ({ ...turn, messages: turn.messages.filter(message => settings.experience === 'advanced' || !message.technical) })).filter(turn => turn.messages.length || turn.activity.length)
    const fallback: ConversationTurn[] = []
    for (const message of snapshot?.messages ?? []) {
      if (settings.experience !== 'advanced' && message.technical) continue
      if (message.role === 'user' || !fallback.length) fallback.push({ id: message.turn_id || `legacy-${message.id}`, messages: [message], activity: [] })
      else fallback[fallback.length - 1].messages.push(message)
    }
    return fallback
  }, [snapshot?.turns, snapshot?.messages, settings.experience])
  const visibleMessages = turns.flatMap(turn => turn.messages)
  const hasWorkContent = Boolean(visibleMessages.length || snapshot?.dashboard?.plan_revision)
  const showPlanCard = Boolean(snapshot?.dashboard?.plan_revision && (
    workflowMode === 'plan' || snapshot?.dashboard?.status === 'awaiting_plan_approval'
  ))
  const actionError = [
    submitInput.error, cancelQueued.error, continueQueue.error, resumeThread.error, killTask.error,
    viewMutation.error, resolveAttention.error, modeMutation.error, planDecision.error,
  ].find(Boolean)

  const toggleInspector = (name: string) => {
    setTerminalOpen(false)
    setInspector(current => current === name ? null : name)
  }

  const patchThread = async (threadId: string, patch: Record<string, unknown>) => {
    await api.patchThread(threadId, patch)
    await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
  }
  const newThread = (projectId?: string) => {
    const id = projectId ?? selectedProject?.id ?? bootstrap?.projects[0]?.id
    if (!id) { setShowProject(true); return }
    createThread.mutate(id)
  }
  const saveSettings = async (patch: Partial<Settings>) => {
    await api.patchSettings(patch)
    await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    if (selectedThreadId) await queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
  }
  const changeAccess = async (policy: 'default' | 'bounded' | 'full' | 'host', token = '') => {
    if (!selectedThreadId || !snapshot) return {}
    const result = await api.changeAccess(selectedThreadId, policy, snapshot.thread.state_revision, token)
    if (!result.requires_confirmation) {
      await queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })
      await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    }
    return result
  }

  if (bootstrapQuery.isLoading) {
    return <div className="boot-screen"><ProjectMark /><span>Opening local workspace</span></div>
  }
  if (bootstrapQuery.isError || !bootstrap) {
    return <div className="boot-screen error"><ProjectMark /><h1>GA3BAD could not start</h1><p>{String(bootstrapQuery.error)}</p><button onClick={() => bootstrapQuery.refetch()}>Try again</button></div>
  }

  return (
    <div className="app-shell" data-workflow={workflowMode}>
      <ProjectSidebar
        bootstrap={bootstrap}
        selectedThreadId={selectedThreadId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={id => { navigate(taskPath(id)); setSidebarOpen(false) }}
        onNewThread={newThread}
        onAddProject={path => path ? addProject.mutate(path) : setShowProject(true)}
        onPatchThread={patchThread}
        onSettings={() => setShowSettings(true)}
      />
      {sidebarOpen && <button className="mobile-scrim" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} />}
      <main className="workspace">
        <header className="workspace-header">
          <div className="header-left">
            <IconButton label="Open sidebar" onClick={() => setSidebarOpen(true)}><Menu size={18} /></IconButton>
            <div className="workspace-identity">
              <strong>{snapshot?.thread.title ?? 'New task'}</strong>
              <span><Folder size={13} />{selectedProject?.name ?? 'No project'}<i /> <GitBranch size={13} />local</span>
            </div>
          </div>
          <div className="header-actions">
            <span className={`work-state ${waitingForAnswer ? 'waiting' : running ? 'running' : ''}`} title="Current task state">{waitingForAnswer ? <CircleHelp size={14} /> : running ? <LoaderCircle className="spin" size={14} /> : <CircleCheck size={14} />}{workState}</span>
            <span className={`connection ${online ? 'online' : ''}`} title="Browser connection to the local backend">{online ? <Wifi size={14} /> : <WifiOff size={14} />}{online ? 'Connected' : 'Reconnecting'}</span>
            {snapshot && <span className={`mode-button mode-${workflowMode}`} title="Workflow mode; change it from the composer Mode menu"><Sparkles size={14} />{workflowMode}</span>}
            <div className="workspace-tools" aria-label="Workspace tools">
              <ToolButton label={snapshot?.thread.view_mode === 'visualize' ? 'Show transcript' : 'Visualize task'} active={snapshot?.thread.view_mode === 'visualize'} disabled={!selectedThreadId} onClick={() => snapshot && viewMutation.mutate(snapshot.thread.view_mode === 'visualize' ? 'transcript' : 'visualize')}><Network size={17} /></ToolButton>
              <ToolButton label="Inspect agents" active={inspector === 'agents'} disabled={!selectedThreadId} onClick={() => toggleInspector('agents')}><Bot size={17} /></ToolButton>
              <ToolButton label="Toggle diff panel" active={inspector === 'changes'} disabled={!selectedThreadId} onClick={() => toggleInspector('changes')}><FileDiff size={17} /></ToolButton>
              <ToolButton label="Toggle terminal" active={terminalOpen} disabled={!snapshot} onClick={() => { setInspector(null); setTerminalOpen(open => !open) }}><SquareTerminal size={17} /></ToolButton>
              <ToolButton label="Explore workspace files" active={inspector === 'files'} disabled={!selectedThreadId} onClick={() => toggleInspector('files')}><FolderTree size={17} /></ToolButton>
            </div>
            <IconButton label="Open plan inspector" disabled={!selectedThreadId} onClick={() => toggleInspector('plan')}><PanelRight size={18} /></IconButton>
            <IconButton label="Settings" onClick={() => setShowSettings(true)}><SettingsIcon size={18} /></IconButton>
          </div>
        </header>
        <div className="resource-overview"><ResourceStrip threadId={selectedThreadId} /></div>

        <div className={`workspace-body ${!hasWorkContent ? 'is-empty' : ''}`}>
          <section className="conversation" aria-live="polite">
            {snapshot?.thread.view_mode === 'visualize' ? (
              visualizationQuery.isError ? <div className="visualize-loading error">Unable to build the task map: {String(visualizationQuery.error)}</div>
                : visualizationQuery.data ? <Suspense fallback={<div className="visualize-loading">Building the durable task map…</div>}><VisualizeView snapshot={visualizationQuery.data as VisualizationSnapshot} /></Suspense>
                  : <div className="visualize-loading"><LoaderCircle className="spin" size={18} />Building the durable task map…</div>
            ) : !selectedThreadId || !hasWorkContent ? <EmptyWorkspace projectName={selectedProject?.name} /> : (
              <div className="transcript-shell"><div className="transcript">
                {turns.map(turn => <section className="conversation-turn" key={turn.id}>{turn.messages.map(message => <MessageView key={message.id} message={message} onDirection={message.role === 'user' ? undefined : (item, direction) => directionMutation.mutate({ message: item, direction })} />)}<TurnActivity turn={turn} onReview={() => toggleInspector('changes')} onUndo={() => { if (window.confirm('Undo the latest accepted checkpoint? The undo remains in Git history.')) submitInput.mutate({ text: '/undo 1' }) }} /></section>)}
                {showPlanCard && snapshot?.dashboard && <PlanCard dashboard={snapshot.dashboard} />}
                <ActivityTimeline presentation={snapshot?.presentation ?? null} startedAt={activeJob?.created_at} />
                {settings.experience === 'advanced' && snapshot?.presentation?.advanced_log?.length ? (
                  <details className="advanced-log"><summary>Technical activity</summary><pre>{snapshot.presentation.advanced_log.join('\n')}</pre></details>
                ) : null}
                <div ref={transcriptEnd} />
              </div><MessageNavigator turns={turns} /></div>
            )}
          </section>

          <div className="interaction-dock">
            {actionError && <div className="action-feedback" role="alert"><CircleAlert size={15} /><span>{String(actionError)}</span></div>}
            {snapshot && ['paused', 'recovery_required', 'problem'].includes(snapshot.thread.status) && (
              <section className="recovery-panel">
                <div><span className="eyebrow">{statusLabel(snapshot.thread.status)}</span><strong>{snapshot.thread.status === 'recovery_required' ? 'Work stopped at an uncertain boundary.' : snapshot.thread.status === 'problem' ? 'This task needs attention before it can continue.' : 'This task is safely paused.'}</strong><small>{snapshot.thread.status === 'recovery_required' ? 'Review the last checkpoint, then resume explicitly. GA3BAD never auto-resumes after a crash.' : snapshot.thread.status === 'problem' ? 'Review the error above, fix its cause, then resume from the durable state.' : 'The durable queue and transcript are preserved.'}</small></div>
                <button disabled={resumeThread.isPending || snapshot.capabilities?.resume?.allowed === false} title={snapshot.capabilities?.resume?.allowed === false ? `${snapshot.capabilities.resume.reason} ${snapshot.capabilities.resume.remediation}` : 'Resume from the durable checkpoint'} onClick={() => resumeThread.mutate()}><Play size={15} />Resume task</button>
              </section>
            )}
            <QueueTray items={snapshot?.queue ?? []} onCancel={jobId => cancelQueued.mutate(jobId)} />
            {snapshot?.queue?.length && !running && snapshot.capabilities?.continue_queue?.allowed ? <button className="continue-queue" disabled={continueQueue.isPending} onClick={() => continueQueue.mutate()}><ListPlus size={14} />Continue queue</button> : null}
            {snapshot?.dashboard?.tasks?.length && snapshot.progress ? <PlanProgress dashboard={snapshot.dashboard} progress={snapshot.progress} changes={changesQuery.data} /> : null}
            {snapshot?.dashboard?.status === 'awaiting_plan_approval' && snapshot.dashboard.plan_fingerprint ? <PlanDecisionPanel dashboard={snapshot.dashboard} busy={planDecision.isPending || (snapshot.queue?.length ?? 0) > 0 || snapshot.capabilities?.implement_plan?.allowed === false} blockedReason={snapshot.capabilities?.implement_plan?.allowed === false ? `${snapshot.capabilities.implement_plan.reason} ${snapshot.capabilities.implement_plan.remediation}` : undefined} onImplement={() => planDecision.mutate({ action: 'implement' })} onRevise={feedback => planDecision.mutate({ action: 'keep_planning', feedback })} /> : null}
            <AnimatePresence>
              {snapshot?.presentation?.attention && (
                <AttentionPanel
                  attention={snapshot.presentation.attention}
                  busy={resolveAttention.isPending}
                  onResolve={(key, text) => resolveAttention.mutate({ key, text })}
                />
              )}
            </AnimatePresence>
            {selectedThreadId ? (
              <Composer
                disabled={submitInput.isPending}
                running={running}
                activeJobId={activeJob?.id}
                value={draftText}
                queueCount={snapshot?.queue?.length ?? 0}
                sendCapability={snapshot?.capabilities?.send}
                modeCapability={snapshot?.capabilities?.change_mode}
                pauseCapability={snapshot?.capabilities?.pause}
                killCapability={snapshot?.capabilities?.kill}
                onValueChange={setDraftText}
                onSubmit={text => submitInput.mutate({ text, delivery: 'queue' })}
                onGuide={text => submitInput.mutate({ text, delivery: 'guidance' })}
                onStop={jobId => { void api.checkpoint(jobId).then(() => queryClient.invalidateQueries({ queryKey: ['thread', selectedThreadId] })) }}
                onKill={() => setShowKillTask(true)}
                showQuickStarts={!hasWorkContent}
                workflowMode={workflowMode}
                access={effectiveAccess}
                onModeChange={mode => modeMutation.mutate(mode)}
                onAccessChange={access => { if (access === 'full') setShowFullAccess(true); else void changeAccess(access) }}
              />
            ) : (
              <button className="add-project-primary" onClick={() => setShowProject(true)}><Plus size={17} />Add a local project</button>
            )}
          </div>
        </div>
      </main>

      <AnimatePresence>{inspector && selectedThreadId && <WorkspaceInspector threadId={selectedThreadId} name={inspector} onClose={() => setInspector(null)} />}</AnimatePresence>
      <AnimatePresence>{terminalOpen && selectedThreadId && <TerminalPanel threadId={selectedThreadId} onClose={() => setTerminalOpen(false)} />}</AnimatePresence>
      {inspector && selectedThreadId && (
        <nav className="inspector-tabs" aria-label="Inspector sections">
          {inspectors.map(([value, label]) => <button key={value} className={inspector === value ? 'active' : ''} onClick={() => setInspector(value)}>{label}</button>)}
        </nav>
      )}
      <AnimatePresence>{showSettings && <SettingsDialog settings={settings} threadId={selectedThreadId} onClose={() => setShowSettings(false)} onSave={saveSettings} />}</AnimatePresence>
      <AnimatePresence>{showFullAccess && <FullAccessDialog onClose={() => setShowFullAccess(false)} onApply={(policy, token) => changeAccess(policy, token)} />}</AnimatePresence>
      <AnimatePresence>{showKillTask && snapshot && <KillTaskDialog title={snapshot.thread.title} busy={killTask.isPending} onClose={() => !killTask.isPending && setShowKillTask(false)} onConfirm={() => killTask.mutate()} />}</AnimatePresence>
      <AnimatePresence>{showProject && <ProjectDialog onClose={() => setShowProject(false)} onSubmit={path => addProject.mutate(path)} busy={addProject.isPending} error={addProject.error ? String(addProject.error) : undefined} />}</AnimatePresence>
    </div>
  )
}
