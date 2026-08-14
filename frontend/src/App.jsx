import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Badge } from './components/Primitives';
import { GraphPanel } from './components/GraphPanel';
import {
  StrategyPanel,
  ConfidencePanel,
  ReflectionPanel,
  CitationPanel,
  MemoryPanel,
  NodeProgressPanel,
} from './components/Panels';
import { ChatMessage, TypingIndicator } from './components/ChatMessage';

const API = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  ? 'http://localhost:8000'
  : 'https://ari-backend-hgnh.onrender.com';

export default function App() {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: "Hello! I'm ARI — Adaptive Retrieval Intelligence. I select the best retrieval strategy for each question. Ask me anything, and watch the pipeline run in real time.",
    meta: null,
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [completedNodes, setCompletedNodes] = useState(new Set());
  const [activeNode, setActiveNode] = useState(null);
  const [nodeEvents, setNodeEvents] = useState([]);
  const [nodeData, setNodeData] = useState({});
  const [strategies, setStrategies] = useState([]);
  const [reasoning, setReasoning] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [reflection, setReflection] = useState(null);
  const [citations, setCitations] = useState([]);
  const [memoryContext, setMemoryContext] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState(null);

  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `🔄 Ingesting document: "${file.name}"... Parsing and indexing content. Please wait.`,
      meta: null
    }]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch(`${API}/ingestion/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const res = await resp.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ Document successfully ingested!\n* **Added chunks**: ${res.chunks_ingested}\n* **Source**: \`${res.filename}\`\n\nYou can now ask questions about the contents of this file! The adaptive RAG pipeline will automatically fetch and cite it.`,
        meta: null
      }]);
    } catch (err) {
      setError(`Ingestion failed: ${err.message}`);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Ingestion failed for "${file.name}".\n\nError: ${err.message}`,
        meta: null
      }]);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const resetPipelineState = () => {
    setCompletedNodes(new Set());
    setActiveNode(null);
    setNodeEvents([]);
    setNodeData({});
    setStrategies([]);
    setReasoning('');
    setConfidence(null);
    setReflection(null);
    setCitations([]);
    setError(null);
  };

  const submit = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput('');
    setLoading(true);
    resetPipelineState();

    setMessages(prev => [...prev, { role: 'user', content: q, meta: null }]);

    let finalAnswer = null;
    let finalMeta = null;

    try {
      const resp = await fetch(`${API}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          conversation_id: conversationId,
        }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          if (event.event === 'node_complete') {
            const node = event.node;
            setActiveNode(node);
            setCompletedNodes(prev => new Set([...prev, node]));
            setNodeEvents(prev => [...prev, event]);
            setNodeData(prev => ({ ...prev, [node]: event.data || {} }));

            if (node === 'adaptive_router') {
              if (event.data?.strategies) setStrategies(event.data.strategies);
            }
            if (node === 'confidence_score') {
              if (event.data?.confidence_level) {
                setConfidence({
                  confidence_level: event.data.confidence_level,
                  confidence_score: event.data.confidence_score,
                });
              }
            }
            if (node === 'reflect') {
              setReflection({
                is_supported: event.data?.is_supported,
                should_retry: event.data?.should_retry,
              });
            }
          }

          if (event.event === 'final') {
            finalAnswer = event.answer || '';
            finalMeta = {
              confidence_level: event.confidence?.confidence_level,
              confidence_score: event.confidence?.confidence_score,
              retry_count: event.retry_count || 0,
              strategies: event.strategies_used || [],
            };
            setConfidence(event.confidence);
            setCitations(event.citations || []);
            setStrategies(event.strategies_used || []);
            if (event.conversation_id) setConversationId(event.conversation_id);
            if (event.memory_context) setMemoryContext(event.memory_context);
          }

          if (event.event === 'error') {
            setError(event.message || 'An error occurred');
            setLoading(false);
            setActiveNode(null);
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
      setActiveNode(null);
      return;
    }

    setActiveNode(null);
    setLoading(false);

    if (finalAnswer !== null) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: finalAnswer || 'I was unable to generate a response.',
        meta: finalMeta,
      }]);
    }
  }, [input, loading, conversationId]);

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  return (
    <>
      {/* ── Header ── */}
      <div style={{
        height: 48, borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', padding: '0 20px',
        gap: 12, background: 'var(--surface)',
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 10,
      }}>
        <span style={{ fontFamily: 'var(--mono)', fontWeight: 500, color: 'var(--accent)', fontSize: 14 }}>⬡ ARI</span>
        <span style={{ color: 'var(--border)' }}>│</span>
        <span style={{ fontSize: 11, color: 'var(--dim)' }}>Adaptive Retrieval Intelligence</span>
        {conversationId && (
          <>
            <span style={{ color: 'var(--border)' }}>│</span>
            <Badge small color='#6b7280'>conv {conversationId.slice(0, 8)}</Badge>
          </>
        )}
        {loading && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)',
              animation: 'pulse 1s ease-in-out infinite'
            }} />
            <span style={{ fontSize: 11, color: 'var(--dim)', fontFamily: 'var(--mono)' }}>
              {activeNode?.replace(/_/g, ' ') || 'processing'}
            </span>
          </div>
        )}
      </div>

      {/* ── Main Layout ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '280px 1fr 240px',
        gridTemplateRows: 'auto',
        gap: 12, padding: '60px 12px 12px',
        height: '100vh', overflow: 'hidden',
      }}>

        {/* ── LEFT COLUMN ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, overflow: 'auto', paddingRight: 2 }}>
          <GraphPanel completedNodes={completedNodes} activeNode={activeNode} />
        </div>

        {/* ── CENTER COLUMN ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, overflow: 'hidden' }}>

          {/* Chat */}
          <div style={{
            flex: 1, background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden',
          }}>
            <div style={{
              padding: '10px 14px', borderBottom: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', gap: 8,
              background: 'linear-gradient(180deg,#1a1e26 0%,var(--surface) 100%)',
            }}>
              <span style={{
                fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
                letterSpacing: '0.1em', color: 'var(--dim)'
              }}>◎ Chat</span>
            </div>
            <div ref={chatRef} style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
              {messages.map((m, i) => <ChatMessage key={i} {...m} />)}
              {loading && <TypingIndicator activeNode={activeNode} />}
              {error && (
                <div style={{
                  padding: '10px 13px', background: '#ef444418',
                  border: '1px solid #ef444440', borderRadius: 8, marginBottom: 12
                }}>
                  <span style={{ color: '#ef4444', fontSize: 12 }}>⚠ {error}</span>
                </div>
              )}
            </div>
            <div style={{
              padding: 12, borderTop: '1px solid var(--border)',
              display: 'flex', gap: 8, background: 'var(--bg)'
            }}>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".txt,.pdf"
                style={{ display: 'none' }}
              />
              <button onClick={() => fileInputRef.current?.click()} disabled={loading || uploading} style={{
                width: 40, background: 'var(--muted)',
                border: 'none', borderRadius: 6, cursor: loading || uploading ? 'not-allowed' : 'pointer',
                color: 'var(--dim)', fontSize: 16, transition: 'all 0.2s',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>📎</button>
              <textarea
                ref={inputRef}
                rows={2}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask anything — e.g. 'What is LangGraph persistence?' or 'Compare vector search vs hybrid search'"
                disabled={loading || uploading}
                style={{
                  flex: 1, background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 6, padding: '8px 10px', color: 'var(--text)',
                  fontSize: 13, fontFamily: 'var(--sans)', lineHeight: 1.5,
                  opacity: loading || uploading ? 0.5 : 1,
                }}
              />
              <button onClick={submit} disabled={loading || uploading || !input.trim()} style={{
                width: 40, background: loading || uploading || !input.trim() ? 'var(--muted)' : 'var(--accent)',
                border: 'none', borderRadius: 6, cursor: loading || uploading ? 'not-allowed' : 'pointer',
                color: '#fff', fontSize: 16, transition: 'all 0.2s',
              }}>↑</button>
            </div>
          </div>

          {/* Bottom row: Node Progress + Citations */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, height: 200 }}>
            <NodeProgressPanel events={nodeEvents} />
            <CitationPanel citations={citations} />
          </div>
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, overflow: 'auto' }}>
          <StrategyPanel strategies={strategies} reasoning={reasoning} nodeData={nodeData} />
          <ConfidencePanel confidence={confidence} />
          <ReflectionPanel reflection={reflection} />
          <MemoryPanel memoryContext={memoryContext} />
        </div>

      </div>
    </>
  );
}
