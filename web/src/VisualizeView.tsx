import { useMemo, useState } from 'react'
import dagre from '@dagrejs/dagre'
import {
  Background, Controls, Handle, MiniMap, Position, ReactFlow,
  type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Bot, CircleAlert, CircleCheck, GitBranch, Search, ShieldQuestion, Sparkles, SquareTerminal, X } from 'lucide-react'
import type { VisualizationNode, VisualizationSnapshot } from './types'

type FlowData = VisualizationNode & Record<string, unknown> & { collapsed?: boolean }
type WorkflowFlowNode = Node<FlowData, 'workflow'>

const icons = {
  agent: Bot,
  error: CircleAlert,
  checkpoint: CircleCheck,
  approval: ShieldQuestion,
  question: ShieldQuestion,
  task: GitBranch,
  tool: SquareTerminal,
  goal: Sparkles,
}

function WorkflowNode({ data, selected }: NodeProps<WorkflowFlowNode>) {
  const Icon = icons[data.kind as keyof typeof icons] ?? CircleCheck
  return (
    <div className={`visual-node visual-node-${data.kind} status-${data.status} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <span className="visual-node-icon"><Icon size={14} /></span>
      <div><strong>{data.label}</strong><small>{data.status.replaceAll('_', ' ')}</small></div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = { workflow: WorkflowNode }

function layout(snapshot: VisualizationSnapshot, hidden: Set<string>) {
  const graph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  graph.setGraph({ rankdir: 'LR', nodesep: 34, ranksep: 82, marginx: 36, marginy: 36 })
  const visible = snapshot.nodes.filter(node => !hidden.has(node.id))
  visible.forEach(node => graph.setNode(node.id, { width: 190, height: 62 }))
  snapshot.edges.filter(edge => !hidden.has(edge.source) && !hidden.has(edge.target)).forEach(edge => graph.setEdge(edge.source, edge.target))
  dagre.layout(graph)
  const nodes: WorkflowFlowNode[] = visible.map(item => {
    const point = graph.node(item.id) ?? { x: 0, y: 0 }
    return { id: item.id, type: 'workflow', position: { x: point.x - 95, y: point.y - 31 }, data: { ...item } }
  })
  const edges: Edge[] = snapshot.edges.filter(edge => !hidden.has(edge.source) && !hidden.has(edge.target)).map(item => ({
    id: item.id, source: item.source, target: item.target, type: 'smoothstep',
    animated: item.kind === 'retry', className: `visual-edge visual-edge-${item.kind}`,
  }))
  return { nodes, edges }
}

export function VisualizeView({ snapshot }: { snapshot: VisualizationSnapshot }) {
  const [selectedId, setSelectedId] = useState<string | null>(snapshot.current_node_id)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const descendants = useMemo(() => {
    const children = new Map<string, string[]>()
    snapshot.nodes.forEach(node => {
      if (!node.parent_id) return
      children.set(node.parent_id, [...(children.get(node.parent_id) ?? []), node.id])
    })
    const hidden = new Set<string>()
    const visit = (id: string) => (children.get(id) ?? []).forEach(child => { hidden.add(child); visit(child) })
    collapsed.forEach(visit)
    return hidden
  }, [snapshot.nodes, collapsed])
  const flow = useMemo(() => layout(snapshot, descendants), [snapshot, descendants])
  const selected = snapshot.nodes.find(node => node.id === selectedId)
  const matches = query.trim() ? snapshot.nodes.filter(node => `${node.label} ${node.summary}`.toLowerCase().includes(query.toLowerCase())).slice(0, 8) : []

  return (
    <section className="visualize-surface" aria-label="Visual task map">
      <div className="visualize-toolbar">
        <label><Search size={14} /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find a step or agent" /></label>
        <span>{snapshot.nodes.length} nodes · live durable state</span>
        {matches.length > 0 && <div className="visual-search-results">{matches.map(node => <button key={node.id} onClick={() => { setSelectedId(node.id); setQuery('') }}>{node.label}<small>{node.status}</small></button>)}</div>}
      </div>
      <ReactFlow
        nodes={flow.nodes}
        edges={flow.edges}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.25}
        maxZoom={1.8}
        nodesDraggable={false}
        nodesConnectable={false}
        onNodeClick={(_, node) => setSelectedId(node.id)}
        onNodeDoubleClick={(_, node) => setCollapsed(value => { const next = new Set(value); next.has(node.id) ? next.delete(node.id) : next.add(node.id); return next })}
      >
        <Background gap={24} size={1} />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable nodeColor={node => node.id === snapshot.current_node_id ? 'var(--mode-accent)' : 'var(--faint)'} />
      </ReactFlow>
      {selected && <aside className="visual-node-detail">
        <button aria-label="Close node details" onClick={() => setSelectedId(null)}><X size={15} /></button>
        <span className="eyebrow">{selected.kind} · {selected.status}</span>
        <h2>{selected.label}</h2>
        {selected.summary && <p>{selected.summary}</p>}
        {selected.details && Object.entries(selected.details).map(([key, value]) => <div className="node-detail-row" key={key}><strong>{key.replaceAll('_', ' ')}</strong><span>{Array.isArray(value) ? value.join(', ') : String(value ?? '—')}</span></div>)}
        <button className="collapse-node" onClick={() => setCollapsed(value => { const next = new Set(value); next.has(selected.id) ? next.delete(selected.id) : next.add(selected.id); return next })}>{collapsed.has(selected.id) ? 'Expand branch' : 'Collapse branch'}</button>
      </aside>}
    </section>
  )
}
