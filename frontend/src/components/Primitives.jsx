import React from 'react';

export const Badge = ({ children, color = '#4f9cf9', small }) => (
  <span style={{
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: small ? '2px 6px' : '3px 8px',
    background: color + '18', border: `1px solid ${color}40`,
    borderRadius: 4, fontSize: small ? 11 : 12, color, fontFamily: 'var(--mono)', fontWeight: 500,
  }}>{children}</span>
);

export const Pill = ({ label, value, color = 'var(--accent)' }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
    <span style={{ fontSize: 10, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
    <span style={{ fontSize: 13, color, fontFamily: 'var(--mono)', fontWeight: 500 }}>{value}</span>
  </div>
);

export const Panel = ({ title, icon, children, style }) => (
  <div style={{
    background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8,
    overflow: 'hidden', display: 'flex', flexDirection: 'column', ...style
  }}>
    <div style={{
      padding: '10px 14px', borderBottom: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', gap: 8,
      background: 'linear-gradient(180deg,#1a1e26 0%,var(--surface) 100%)',
    }}>
      {icon && <span style={{ fontSize: 13, opacity: 0.7 }}>{icon}</span>}
      <span style={{
        fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '0.1em', color: 'var(--dim)'
      }}>{title}</span>
    </div>
    <div style={{ flex: 1, overflow: 'auto', padding: 14 }}>{children}</div>
  </div>
);

export const Meter = ({ label, value, color = 'var(--accent)' }) => (
  <div style={{ marginBottom: 10 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ fontSize: 11, color: 'var(--dim)' }}>{label}</span>
      <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color }}>{(value * 100).toFixed(0)}%</span>
    </div>
    <div style={{ height: 3, background: 'var(--muted)', borderRadius: 2 }}>
      <div style={{
        height: '100%', width: `${value * 100}%`, background: color,
        borderRadius: 2, transition: 'width 0.6s ease'
      }} />
    </div>
  </div>
);
