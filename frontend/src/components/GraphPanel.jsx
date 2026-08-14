import React from 'react';
import { Panel } from './Primitives';

const GRAPH_NODES = [
  { id: 'memory_load', label: 'Memory Load', x: 200, y: 30 },
  { id: 'query_understanding', label: 'Query Understanding', x: 200, y: 100 },
  { id: 'adaptive_router', label: 'Adaptive Router', x: 200, y: 170 },
  { id: 'retrieve', label: 'Retrieve', x: 80, y: 250 },
  { id: 'document_merge', label: 'Doc Merge', x: 80, y: 320 },
  { id: 'rerank', label: 'Rerank', x: 80, y: 390 },
  { id: 'context_validate', label: 'Validate Context', x: 80, y: 460 },
  { id: 'generate', label: 'Generate', x: 200, y: 540 },
  { id: 'reflect', label: 'Reflect', x: 200, y: 610 },
  { id: 'confidence_score', label: 'Confidence', x: 200, y: 680 },
  { id: 'query_rewrite', label: 'Query Rewrite', x: 340, y: 430 },
  { id: 'finalize', label: 'Finalize', x: 200, y: 750 },
  { id: 'memory_save', label: 'Memory Save', x: 200, y: 820 },
];

const GRAPH_EDGES = [
  ['memory_load', 'query_understanding'], ['query_understanding', 'adaptive_router'],
  ['adaptive_router', 'retrieve'], ['retrieve', 'document_merge'],
  ['document_merge', 'rerank'], ['rerank', 'context_validate'],
  ['context_validate', 'generate'], ['generate', 'reflect'],
  ['reflect', 'confidence_score'], ['confidence_score', 'finalize'],
  ['finalize', 'memory_save'], ['context_validate', 'query_rewrite'],
  ['confidence_score', 'query_rewrite'], ['query_rewrite', 'query_understanding'],
  ['adaptive_router', 'finalize'],
];

export const GraphPanel = ({ completedNodes, activeNode }) => {
  const nodeMap = Object.fromEntries(GRAPH_NODES.map(n => [n.id, n]));
  const W = 420, H = 880;
  return (
    <Panel title="Graph Execution" icon="◈" style={{ height: '100%' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <defs>
          <linearGradient id="activeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#059669" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
          </linearGradient>
          <linearGradient id="doneGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#047857" stopOpacity="0.02" />
          </linearGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        {GRAPH_EDGES.map(([a, b], i) => {
          const na = nodeMap[a], nb = nodeMap[b];
          if (!na || !nb) return null;

          const isDone = completedNodes.has(a) && completedNodes.has(b);
          const isActive = (completedNodes.has(a) || activeNode === a) && activeNode === b;

          const startX = na.x + 60;
          const startY = na.y + 28;
          const endX = nb.x + 60;
          const endY = nb.y;

          const midY = (startY + endY) / 2;
          const pathData = `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;

          let stroke = '#1e2228';
          let strokeWidth = 1.2;
          let strokeDasharray = '4 3';
          let animation = 'none';

          if (isActive) {
            stroke = '#10b981';
            strokeWidth = 2;
            strokeDasharray = '6 4';
            animation = 'flowEdge 0.6s linear infinite';
          } else if (isDone) {
            stroke = '#10b98160';
            strokeWidth = 1.5;
            strokeDasharray = 'none';
          }

          return (
            <path key={i} d={pathData} fill="none"
              stroke={stroke} strokeWidth={strokeWidth} strokeDasharray={strokeDasharray}
              style={{ animation, transition: 'all 0.3s' }} />
          );
        })}
        {GRAPH_NODES.map(n => {
          const isActive = activeNode === n.id;
          const isDone = completedNodes.has(n.id);

          let stroke = '#2a2f38';
          let fill = 'rgba(20, 23, 28, 0.6)';
          let strokeWidth = 1;
          let filter = 'none';
          let textWeight = 'normal';
          let textColor = '#6b7280';
          let indicatorColor = 'none';

          if (isActive) {
            stroke = '#10b981';
            fill = 'url(#activeGrad)';
            strokeWidth = 2;
            filter = 'url(#glow)';
            textWeight = '600';
            textColor = '#ffffff';
            indicatorColor = '#10b981';
          } else if (isDone) {
            stroke = '#10b98180';
            fill = 'url(#doneGrad)';
            strokeWidth = 1;
            textColor = '#e2e6ed';
            indicatorColor = '#059669';
          }

          return (
            <g key={n.id} style={{ transition: 'all 0.3s' }}>
              <rect x={n.x} y={n.y} width={120} height={28} rx={5}
                fill={fill} stroke={stroke} strokeWidth={strokeWidth}
                style={{
                  transition: 'all 0.3s ease',
                  animation: isActive ? 'pulseRect 2s ease-in-out infinite' : 'none',
                }} />
              <text x={n.x + 60} y={n.y + 18} textAnchor="middle"
                fontSize={10} fill={textColor} fontFamily="var(--mono)" fontWeight={textWeight}>
                {n.label}
              </text>
              {indicatorColor !== 'none' && (
                <circle cx={n.x + 8} cy={n.y + 14} r={isActive ? 3 : 2} fill={indicatorColor}
                  style={{
                    animation: isActive ? 'pulseOpacity 1s ease-in-out infinite' : 'none'
                  }} />
              )}
            </g>
          );
        })}
      </svg>
    </Panel>
  );
};
