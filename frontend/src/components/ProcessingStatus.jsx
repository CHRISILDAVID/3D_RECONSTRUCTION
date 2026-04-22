import { useState, useEffect, useRef } from 'react'

const PIPELINE_STAGES = [
    { key: 'uploading', label: 'Uploading', icon: '📤' },
    { key: 'loading_model', label: 'Loading Model', icon: '🧠' },
    { key: 'loading_images', label: 'Loading Images', icon: '🖼️' },
    { key: 'matching', label: 'Stereo Matching', icon: '🔗' },
    { key: 'aligning', label: 'Global Alignment', icon: '🌍' },
    { key: 'extracting', label: 'Extracting 3D', icon: '📐' },
    { key: 'exporting', label: 'Exporting', icon: '💾' },
]

export default function ProcessingStatus({ sessionId, images }) {
    const [progress, setProgress] = useState(5)
    const [message, setMessage] = useState('Starting reconstruction...')
    const [currentStage, setCurrentStage] = useState('uploading')
    const wsRef = useRef(null)
    const progressInterval = useRef(null)

    useEffect(() => {
        if (!sessionId) return

        // Connect WebSocket for progress
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/progress/${sessionId}`

        try {
            const ws = new WebSocket(wsUrl)
            wsRef.current = ws

            ws.onopen = () => {
                console.log('WebSocket connected')
            }

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data)
                if (data.type === 'progress') {
                    setProgress(data.progress)
                    setMessage(data.message)
                    setCurrentStage(data.stage)
                }
            }

            ws.onerror = () => {
                console.log('WebSocket error, using polling fallback')
            }

            ws.onclose = () => {
                console.log('WebSocket closed')
            }
        } catch {
            console.log('WebSocket unavailable')
        }

        // Simulated progress increment for visual feedback
        progressInterval.current = setInterval(() => {
            setProgress(prev => {
                if (prev >= 95) return prev
                return prev + 0.3
            })
        }, 500)

        return () => {
            if (wsRef.current) wsRef.current.close()
            if (progressInterval.current) clearInterval(progressInterval.current)
        }
    }, [sessionId])

    const currentStageIndex = PIPELINE_STAGES.findIndex(s => currentStage.startsWith(s.key))

    return (
        <div className="processing-section">
            <div className="processing-card">
                {/* Header */}
                <div className="processing-header">
                    <div className="processing-spinner-lg">
                        <svg viewBox="0 0 50 50" className="circular-progress">
                            <circle className="path-bg" cx="25" cy="25" r="20" />
                            <circle
                                className="path"
                                cx="25" cy="25" r="20"
                                strokeDasharray={`${progress * 1.256} 125.6`}
                            />
                        </svg>
                        <span className="progress-text">{Math.round(progress)}%</span>
                    </div>
                    <div>
                        <h2 className="processing-title">Reconstructing Scene</h2>
                        <p className="processing-message">{message}</p>
                    </div>
                </div>

                {/* Progress bar */}
                <div className="progress-bar-container">
                    <div className="progress-bar">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>

                {/* Pipeline stages */}
                <div className="pipeline-stages">
                    {PIPELINE_STAGES.map((stage, idx) => {
                        const isActive = idx === currentStageIndex
                        const isDone = idx < currentStageIndex
                        return (
                            <div
                                key={stage.key}
                                className={`pipeline-stage ${isActive ? 'stage-active' : ''} ${isDone ? 'stage-done' : ''}`}
                            >
                                <div className="stage-indicator">
                                    {isDone ? (
                                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                                            <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z" />
                                        </svg>
                                    ) : isActive ? (
                                        <span className="stage-dot pulse" />
                                    ) : (
                                        <span className="stage-dot" />
                                    )}
                                </div>
                                <div className="stage-info">
                                    <span className="stage-icon">{stage.icon}</span>
                                    <span className="stage-label">{stage.label}</span>
                                </div>
                            </div>
                        )
                    })}
                </div>

                {/* Image thumbnails */}
                {images.length > 0 && (
                    <div className="processing-images">
                        <p className="processing-images-label">Input Images</p>
                        <div className="processing-images-grid">
                            {images.map((img, idx) => (
                                <div key={idx} className="processing-thumb">
                                    <img src={img.preview} alt={img.name} />
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
