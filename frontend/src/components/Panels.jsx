import React from 'react';
import { Badge, Pill, Panel, Meter } from './Primitives';

export const STRATEGY_META = {
  vector_search: { label: 'Vector Search', color: '#4f9cf9', icon: '⬡' },
  hybrid_search: { label: 'Hybrid Search', color: '#7c3aed', icon: '⟐' },
  multi_query_retrieval: { label: 'Multi-Query', color: '#10b981', icon: '⊕' },
  web_search: { label: 'Web Search', color: '#f59e0b', icon: '⊛' },
  graph_rag: { label: 'Graph RAG', color: '#ec4899', icon: '⋈' },
};

export const StrategyPanel = ({ strategies, reasoning, nodeData }) => (
  <Panel title="Retrieval Strategy" icon="⟐">
    {strategies.length === 0
      ? <p style={{ color: 'var(--dim)', fontSize: 12 }}>Waiting for router decision…</p>
      : <>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 10 }}>
          {strategies.map(s => {
            const m = STRATEGY_META[s] || { label: s, color: '#6b7280', icon: '○' };
            return <Badge key={s} color={m.color}>{m.icon} {m.label}</Badge>;
          })}
        </div>
        {reasoning && <p style={{ fontSize: 12, color: 'var(--dim)', fontStyle: 'italic' }}>{reasoning}</p>}
        {nodeData.adaptive_router?.clarification_needed && (
          <div style={{
            marginTop: 8, padding: '8px 10px',
            background: '#f59e0b18', border: '1px solid #f59e0b40', borderRadius: 6
          }}>
            <span style={{ color: '#f59e0b', fontSize: 12 }}>⚠ Clarification requested</span>
          </div>
        )}
      </>
    }
  </Panel>
);

export const ConfidencePanel = ({ confidence }) => {
  if (!confidence) return (
    <Panel title="Confidence" icon="◎">
      <p style={{ color: 'var(--dim)', fontSize: 12 }}>Awaiting confidence estimation…</p>
    </Panel>
  );
  const levelColor = confidence.confidence_level === 'high' ? '#10b981'
    : confidence.confidence_level === 'medium' ? '#f59e0b' : '#ef4444';
  return (
    <Panel title="Confidence" icon="◎">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <div style={{
          width: 56, height: 56, borderRadius: '50%', border: `3px solid ${levelColor}`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          background: levelColor + '12',
        }}>
          <span style={{ fontFamily: 'var(--mono)', fontWeight: 600, fontSize: 16, color: levelColor }}>
            {Math.round((confidence.confidence_score || 0) * 100)}
          </span>
          <span style={{ fontSize: 9, color: 'var(--dim)', textTransform: 'uppercase' }}>score</span>
        </div>
        <div>
          <Badge color={levelColor}>{(confidence.confidence_level || 'low').toUpperCase()}</Badge>
          {confidence.reason && (
            <p style={{ fontSize: 11, color: 'var(--dim)', marginTop: 4, maxWidth: 180 }}>{confidence.reason}</p>
          )}
        </div>
      </div>
      <Meter label="Retrieval Quality" value={confidence.retrieval_quality || 0} color='#4f9cf9' />
      <Meter label="Reflection Score" value={confidence.reflection_score || 0} color='#7c3aed' />
      <Meter label="Citation Quality" value={confidence.citation_quality || 0} color='#10b981' />
      <Meter label="Hallucination Risk" value={confidence.hallucination_risk || 0} color='#ef4444' />
      <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border)' }}>
        <Pill label="Sources" value={`${confidence.num_sources || 0} cited`} color='var(--dim)' />
      </div>
    </Panel>
  );
};

export const ReflectionPanel = ({ reflection }) => {
  if (!reflection) return (
    <Panel title="Reflection" icon="⟳">
      <p style={{ color: 'var(--dim)', fontSize: 12 }}>Awaiting reflection agent…</p>
    </Panel>
  );
  return (
    <Panel title="Reflection" icon="⟳">
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <Badge color={reflection.is_supported ? '#10b981' : '#ef4444'}>
          {reflection.is_supported ? '✓ Supported' : '✗ Unsupported'}
        </Badge>
        <Badge color={reflection.should_retry ? '#f59e0b' : '#10b981'}>
          {reflection.should_retry ? '↺ Retry' : '✓ Accept'}
        </Badge>
      </div>
      {reflection.hallucinations?.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <p style={{ fontSize: 11, color: '#ef4444', fontWeight: 500, marginBottom: 4 }}>Hallucinations detected:</p>
          {reflection.hallucinations.map((h, i) => (
            <p key={i} style={{
              fontSize: 11, color: 'var(--dim)', paddingLeft: 8,
              borderLeft: '2px solid #ef444440', marginBottom: 3
            }}>— {h}</p>
          ))}
        </div>
      )}
      {reflection.missing_information && (
        <div>
          <p style={{ fontSize: 11, color: '#f59e0b', fontWeight: 500, marginBottom: 4 }}>Missing information:</p>
          <p style={{ fontSize: 11, color: 'var(--dim)' }}>{reflection.missing_information}</p>
        </div>
      )}
      {!reflection.hallucinations?.length && !reflection.missing_information && (
        <p style={{ fontSize: 12, color: '#10b981' }}>Answer verified — no issues found.</p>
      )}
    </Panel>
  );
};

export const CitationPanel = ({ citations }) => (
  <Panel title="Citations" icon="⟨⟩">
    {citations.length === 0
      ? <p style={{ color: 'var(--dim)', fontSize: 12 }}>No citations yet.</p>
      : citations.map((c, i) => (
        <div key={i} style={{
          display: 'flex', gap: 8, alignItems: 'flex-start',
          marginBottom: 8, padding: '6px 8px',
          background: 'var(--muted)', borderRadius: 5,
        }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--accent)',
            marginTop: 1, flexShrink: 0
          }}>[{i + 1}]</span>
          <span style={{ fontSize: 11, color: 'var(--dim)', wordBreak: 'break-all' }}>{c}</span>
        </div>
      ))
    }
  </Panel>
);

export const MemoryPanel = ({ memoryContext }) => {
  const history = memoryContext?.conversation_history || [];
  const prefs = memoryContext?.user_preferences;
  const cached = memoryContext?.cached_rewrite;
  const hasAny = history.length > 0 || prefs || cached;
  return (
    <Panel title="Memory" icon="◷">
      {!hasAny
        ? <p style={{ color: 'var(--dim)', fontSize: 12 }}>No memory loaded for this session.</p>
        : <>
          {history.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <p style={{
                fontSize: 10, color: 'var(--dim)', textTransform: 'uppercase',
                letterSpacing: '0.08em', marginBottom: 8
              }}>Recent turns</p>
              {history.slice(-3).map((t, i) => (
                <div key={i} style={{
                  marginBottom: 8, padding: '6px 8px',
                  background: 'var(--muted)', borderRadius: 5
                }}>
                  <p style={{ fontSize: 11, color: 'var(--accent)', marginBottom: 2 }}>Q: {t.question}</p>
                  <p style={{ fontSize: 10, color: 'var(--dim)' }}>
                    {t.answer?.slice(0, 100)}{t.answer?.length > 100 ? '…' : ''}
                  </p>
                  <div style={{ marginTop: 4 }}>
                    <Badge small color='#4f9cf9'>conf {(t.confidence || 0).toFixed(2)}</Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
          {prefs && (
            <div style={{ marginBottom: 12 }}>
              <p style={{
                fontSize: 10, color: 'var(--dim)', textTransform: 'uppercase',
                letterSpacing: '0.08em', marginBottom: 6
              }}>Inferred preferences</p>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {prefs.preferred_strategies?.map(s => (
                  <Badge key={s} small color='#7c3aed'>{s}</Badge>
                ))}
                {prefs.preferred_domains?.map(d => (
                  <Badge key={d} small color='#10b981'>{d}</Badge>
                ))}
              </div>
            </div>
          )}
          {cached && (
            <div>
              <p style={{
                fontSize: 10, color: 'var(--dim)', textTransform: 'uppercase',
                letterSpacing: '0.08em', marginBottom: 6
              }}>Cached rewrite</p>
              <div style={{ padding: '6px 8px', background: 'var(--muted)', borderRadius: 5 }}>
                <p style={{ fontSize: 11, color: '#10b981' }}>↪ {cached.successful_rewrite}</p>
                <p style={{ fontSize: 10, color: 'var(--dim)', marginTop: 2 }}>
                  used {cached.success_count}× · {cached.strategies_that_worked?.join(', ')}
                </p>
              </div>
            </div>
          )}
        </>
      }
    </Panel>
  );
};

export const NodeProgressPanel = ({ events }) => (
  <Panel title="Node Progress" icon="▸">
    {events.length === 0
      ? <p style={{ color: 'var(--dim)', fontSize: 12 }}>Pipeline not started.</p>
      : events.map((e, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: 10, color: '#10b981', flexShrink: 0 }}>✓</span>
          <span style={{ fontSize: 11, color: 'var(--text)', fontFamily: 'var(--mono)' }}>
            {e.label || e.node}
          </span>
          {e.data?.strategies && (
            <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              {e.data.strategies.map(s => (
                <Badge key={s} small color={(STRATEGY_META[s] || {}).color || '#6b7280'}>
                  {(STRATEGY_META[s] || { icon: '○' }).icon} {s}
                </Badge>
              ))}
            </div>
          )}
        </div>
      ))
    }
  </Panel>
);
