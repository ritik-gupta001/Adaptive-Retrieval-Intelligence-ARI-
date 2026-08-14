import React from 'react';
import { Badge } from './Primitives';
import { STRATEGY_META } from './Panels';

export const ChatMessage = ({ role, content, meta }) => (
  <div style={{
    display: 'flex', gap: 10, marginBottom: 16,
    flexDirection: role === 'user' ? 'row-reverse' : 'row',
    alignItems: 'flex-start',
  }}>
    <div style={{
      width: 28, height: 28, borderRadius: 6, flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12,
      background: role === 'user' ? 'var(--accent2)' : 'var(--muted)',
      border: `1px solid ${role === 'user' ? '#7c3aed40' : 'var(--border)'}`,
    }}>
      {role === 'user' ? '◎' : '⬡'}
    </div>
    <div style={{ maxWidth: '78%' }}>
      <div style={{
        padding: '10px 13px', borderRadius: 8,
        background: role === 'user' ? '#7c3aed18' : 'var(--surface)',
        border: `1px solid ${role === 'user' ? '#7c3aed40' : 'var(--border)'}`,
        fontSize: 13, lineHeight: 1.65, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {content}
      </div>
      {meta && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 5 }}>
          {meta.confidence_level && (
            <Badge small color={meta.confidence_level === 'high' ? '#10b981' : meta.confidence_level === 'medium' ? '#f59e0b' : '#ef4444'}>
              {meta.confidence_level} confidence
            </Badge>
          )}
          {meta.retry_count > 0 && (
            <Badge small color='#f59e0b'>↺ {meta.retry_count} retr{meta.retry_count === 1 ? 'y' : 'ies'}</Badge>
          )}
          {meta.strategies?.map(s => (
            <Badge key={s} small color={(STRATEGY_META[s] || { color: '#6b7280' }).color}>
              {(STRATEGY_META[s] || { icon: '○' }).icon} {s}
            </Badge>
          ))}
        </div>
      )}
    </div>
  </div>
);

export const TypingIndicator = ({ activeNode }) => (
  <div style={{ display: 'flex', gap: 10, marginBottom: 16, alignItems: 'flex-start' }}>
    <div style={{
      width: 28, height: 28, borderRadius: 6, flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12,
      background: 'var(--muted)', border: '1px solid var(--border)'
    }}>⬡</div>
    <div style={{
      padding: '10px 13px', background: 'var(--surface)',
      border: '1px solid var(--border)', borderRadius: 8
    }}>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        {[0, 150, 300].map(delay => (
          <div key={delay} style={{
            width: 5, height: 5, borderRadius: '50%', background: 'var(--dim)',
            animation: `pulse 1.2s ease-in-out ${delay}ms infinite`
          }} />
        ))}
        {activeNode && (
          <span style={{ fontSize: 11, color: 'var(--dim)', marginLeft: 6, fontFamily: 'var(--mono)' }}>
            {activeNode.replace(/_/g, ' ')}
          </span>
        )}
      </div>
    </div>
  </div>
);
