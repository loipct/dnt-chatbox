import React, { useState } from 'react';
import './styles.css';

const App = () => {
  const [query, setQuery] = useState('');
  const [top_k, setTopK] = useState(0);
  const [messages, setMessages] = useState([]);
  const [resources, setResources] = useState([]);
  const [queryMode, setQueryMode] = useState('Auto');
  const [rerankMode, setRerankMode] = useState('false');
  const [ragMode, setRagMode] = useState('Normal-RAG'); // Thêm state cho chế độ RAG

  const performAction = async () => {
    if (query.trim() === '') return;
  
    // Kiểm tra nếu top_k là số nguyên dương
    const topKInt = parseInt(top_k, 10);
    if (isNaN(topKInt) || topKInt <= 0) {
      console.error('top_k is not positive integer');
      return;
    }

    addMessage(query, 'user');

  
    try {
      let url;
      if (ragMode === 'Self-RAG') {
        // Tạo URL cho Self-RAG
        url = `http://127.0.0.1:8000/search/self_rag/${query}?top_k=${top_k}`;
      } else {
        // Tạo URL cho Normal-RAG
        url = `http://127.0.0.1:8000/search/adaptive_query/${query}?k=${top_k}&rerank_mode=${rerankMode}&query_category=${queryMode}`;
      }
  
      const response = await fetch(url);
      const data = await response.json();
  
      addMessage(data.text, 'bot');
      displayResources(data.ResourceCollection);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  
    setQuery('');
  };
  

  const addMessage = (text, sender) => {
    setMessages(prevMessages => [...prevMessages, { text, sender }]);
  };

  const displayResources = resources => {
    if (!Array.isArray(resources)) {
      console.error('Expected an array for resources, got:', resources);
      return;
    }

    setResources(resources);
  };

  const handleKeyPress = event => {
    if (event.key === 'Enter') {
      performAction();
    }
  };

  return (
    <div className="main-container">
      <div className="chat-container">

        <div className="chat-header">
          <h1>Chat Interface</h1>
        </div>

        <div id="chatbox" className="chatbox">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.text}
            </div>
          ))}
        </div>

        <div className="chat-input-container" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div style={{ flex: '8 1 0' }}>
            <input
              id="query"
              type="text"
              placeholder="Type your query..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              style={{ width: '100%' }}
            />
          </div>
          <div style={{ flex: '2 1 0' }}>
            <input
              id="k"
              type="number"
              placeholder="Type your top_k..."
              value={top_k}
              onChange={e => setTopK(e.target.value)}
              onKeyDown={handleKeyPress}
              style={{ width: '100%' }}
            />
          </div>
          <button onClick={performAction}>Send</button>
        </div>

        <div className="mode-selector">
          <label htmlFor="RAGMode">RAG Mode:</label>
          <select
            id="ragMode"
            value={ragMode}
            onChange={e => setRagMode(e.target.value)}
          >
            <option value="Normal-RAG">Normal-RAG</option>
            <option value="Self-RAG">Self-RAG</option>
          </select>
        </div>

        {ragMode === 'Normal-RAG' && (
          <>
            <div className="mode-selector">
              <label htmlFor="QueryMode">QueryMode:</label>
              <select id="queryMode" value={queryMode} onChange={e => setQueryMode(e.target.value)}>
                <option value="Auto">Auto</option>
                <option value="Factual">Factual</option>
                <option value="Analytical">Analytical</option>
              </select>
            </div>

            <div className="mode-selector">
              <label htmlFor="RerankMode">Rerank:</label>
              <select id="rerankMode" value={rerankMode} onChange={e => setRerankMode(e.target.value)}>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>
          </>
        )}

      </div>

      <div className="resource-collection">
        <h2>Resource Collection</h2>
        <div id="resources">
          <div className="resource-count">
            Number of resources: {resources.length}
          </div>
          {resources.map((resource, index) => (
            <div key={index} className="resource-item">
              <strong>{resource.topic}</strong><br />
              <em>{resource.title}</em><br />
              {resource.principle}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
