import { useState, useCallback } from 'react'
import ImageUploader from './components/ImageUploader'
import ProcessingStatus from './components/ProcessingStatus'
import PointCloudViewer from './components/PointCloudViewer'

const API_BASE = '/api'

const STAGES = {
    UPLOAD: 'upload',
    PROCESSING: 'processing',
    VIEWING: 'viewing',
}

function App() {
    const [stage, setStage] = useState(STAGES.UPLOAD)
    const [sessionId, setSessionId] = useState(null)
    const [error, setError] = useState(null)
    const [pointCloudData, setPointCloudData] = useState(null)
    const [cameraData, setCameraData] = useState(null)
    const [uploadedImages, setUploadedImages] = useState([])
    const [segment, setSegment] = useState(false)

    const handleUpload = useCallback(async (files) => {
        setError(null)
        const formData = new FormData()
        files.forEach(file => formData.append('files', file))

        try {
            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Upload failed')
            }

            const data = await res.json()
            setSessionId(data.session_id)
            setUploadedImages(files.map(f => ({ name: f.name, preview: URL.createObjectURL(f) })))
            setStage(STAGES.PROCESSING)

            // Trigger reconstruction
            startReconstruction(data.session_id, segment)
        } catch (err) {
            setError(err.message)
        }
    }, [segment])

    const startReconstruction = async (sid, enableSegment = false) => {
        try {
            const url = enableSegment
                ? `${API_BASE}/reconstruct/${sid}?segment=true`
                : `${API_BASE}/reconstruct/${sid}`
            const res = await fetch(url, { method: 'POST' })
            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Reconstruction failed')
            }

            // Fetch results
            const [pcRes, camRes] = await Promise.all([
                fetch(`${API_BASE}/result/${sid}/pointcloud-json`),
                fetch(`${API_BASE}/result/${sid}/cameras`),
            ])

            if (!pcRes.ok || !camRes.ok) throw new Error('Failed to fetch results')

            const pcData = await pcRes.json()
            const camData = await camRes.json()

            setPointCloudData(pcData)
            setCameraData(camData)
            setStage(STAGES.VIEWING)
        } catch (err) {
            setError(err.message)
            setStage(STAGES.UPLOAD)
        }
    }

    const handleReset = () => {
        setStage(STAGES.UPLOAD)
        setSessionId(null)
        setError(null)
        setPointCloudData(null)
        setCameraData(null)
        setUploadedImages([])
    }

    return (
        <div className="app">
            {/* Animated background */}
            <div className="bg-grid" />
            <div className="bg-glow bg-glow-1" />
            <div className="bg-glow bg-glow-2" />
            <div className="bg-glow bg-glow-3" />

            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">
                            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                                <path d="M16 2L28 9V23L16 30L4 23V9L16 2Z" stroke="url(#grad)" strokeWidth="2" fill="none" />
                                <path d="M16 2L28 9L16 16L4 9L16 2Z" fill="url(#grad)" opacity="0.3" />
                                <path d="M16 16V30" stroke="url(#grad)" strokeWidth="1.5" />
                                <path d="M4 9L16 16L28 9" stroke="url(#grad)" strokeWidth="1.5" />
                                <circle cx="16" cy="10" r="2" fill="url(#grad)" />
                                <circle cx="10" cy="20" r="1.5" fill="url(#grad)" opacity="0.7" />
                                <circle cx="22" cy="18" r="1.5" fill="url(#grad)" opacity="0.7" />
                                <defs>
                                    <linearGradient id="grad" x1="4" y1="2" x2="28" y2="30">
                                        <stop stopColor="#818cf8" />
                                        <stop offset="1" stopColor="#c084fc" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <div>
                            <h1 className="logo-text">ReconstructAI</h1>
                            <p className="logo-sub">3D Scene Reconstruction</p>
                        </div>
                    </div>

                    {stage === STAGES.VIEWING && (
                        <button className="btn btn-outline" onClick={handleReset}>
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                                <path d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.418A6 6 0 1 1 8 2v1z" />
                                <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z" />
                            </svg>
                            New Reconstruction
                        </button>
                    )}
                </div>
            </header>

            {/* Main content */}
            <main className="main">
                {error && (
                    <div className="error-banner">
                        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                        <span>{error}</span>
                        <button className="error-close" onClick={() => setError(null)}>×</button>
                    </div>
                )}

                {stage === STAGES.UPLOAD && (
                    <ImageUploader
                        onUpload={handleUpload}
                        segment={segment}
                        onSegmentChange={setSegment}
                    />
                )}

                {stage === STAGES.PROCESSING && (
                    <ProcessingStatus
                        sessionId={sessionId}
                        images={uploadedImages}
                    />
                )}

                {stage === STAGES.VIEWING && (
                    <PointCloudViewer
                        pointCloudData={pointCloudData}
                        cameraData={cameraData}
                        sessionId={sessionId}
                    />
                )}
            </main>

            {/* Footer */}
            <footer className="footer">
                <p>Powered by DUSt3R · Inspired by IMC 2025 1st Place Solution</p>
            </footer>
        </div>
    )
}

export default App
