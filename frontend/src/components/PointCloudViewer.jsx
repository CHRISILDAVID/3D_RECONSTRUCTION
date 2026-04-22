import { useMemo, useState, useRef, useEffect } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Stats } from '@react-three/drei'
import * as THREE from 'three'
import CameraFrustum from './CameraFrustum'

/**
 * Point cloud mesh rendered from position + color arrays
 */
function PointCloud({ positions, colors, pointSize = 0.05 }) {
    const pointsRef = useRef()

    const [geometry, material] = useMemo(() => {
        const geo = new THREE.BufferGeometry()
        const posArray = new Float32Array(positions)
        const colArray = new Float32Array(colors)

        geo.setAttribute('position', new THREE.Float32BufferAttribute(posArray, 3))
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colArray, 3))
        geo.computeBoundingSphere()

        const mat = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
        })

        return [geo, mat]
    }, [positions, colors, pointSize])

    return <points ref={pointsRef} geometry={geometry} material={material} />
}

/**
 * Auto-centers and auto-scales the scene on mount
 */
function SceneSetup({ positions }) {
    const { camera } = useThree()

    useEffect(() => {
        if (!positions || positions.length < 3) return

        // Compute bounding box
        const posArray = new Float32Array(positions)
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(posArray, 3))
        geo.computeBoundingBox()
        geo.computeBoundingSphere()

        const center = new THREE.Vector3()
        geo.boundingBox.getCenter(center)
        const radius = geo.boundingSphere.radius

        // Position camera
        camera.position.set(center.x + radius * 1.5, center.y + radius * 0.8, center.z + radius * 1.5)
        camera.lookAt(center)
        camera.near = radius * 0.01
        camera.far = radius * 100
        camera.updateProjectionMatrix()

        geo.dispose()
    }, [positions, camera])

    return null
}

/**
 * Floating particles background for atmosphere
 */
function FloatingParticles() {
    const ref = useRef()
    const count = 200

    const positions = useMemo(() => {
        const arr = new Float32Array(count * 3)
        for (let i = 0; i < count; i++) {
            arr[i * 3] = (Math.random() - 0.5) * 100
            arr[i * 3 + 1] = (Math.random() - 0.5) * 100
            arr[i * 3 + 2] = (Math.random() - 0.5) * 100
        }
        return arr
    }, [])

    useFrame((state) => {
        if (ref.current) {
            ref.current.rotation.y = state.clock.elapsedTime * 0.01
        }
    })

    return (
        <points ref={ref}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    array={positions}
                    count={count}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial size={0.15} color="#4338ca" transparent opacity={0.3} />
        </points>
    )
}

/**
 * Main Point Cloud Viewer component
 */
export default function PointCloudViewer({ pointCloudData, cameraData, sessionId }) {
    const [pointSize, setPointSize] = useState(0.05)
    const [showCameras, setShowCameras] = useState(true)
    const [showAxes, setShowAxes] = useState(true)
    const [bgColor, setBgColor] = useState('#0a0a1a')
    const [showStats, setShowStats] = useState(false)

    if (!pointCloudData) return null

    const { positions, colors, n_points } = pointCloudData

    return (
        <div className="viewer-section">
            {/* Controls panel */}
            <div className="viewer-controls">
                <div className="controls-group">
                    <h3 className="controls-title">Scene Info</h3>
                    <div className="stats-row">
                        <span className="stat-label">Points</span>
                        <span className="stat-value">{n_points.toLocaleString()}</span>
                    </div>
                    <div className="stats-row">
                        <span className="stat-label">Cameras</span>
                        <span className="stat-value">{cameraData?.length || 0}</span>
                    </div>
                </div>

                <div className="controls-group">
                    <h3 className="controls-title">Display</h3>

                    <div className="control-row">
                        <label className="control-label">Point Size</label>
                        <div className="slider-row">
                            <input
                                type="range"
                                min="0.01"
                                max="1.0"
                                step="0.01"
                                value={pointSize}
                                onChange={(e) => setPointSize(Number(e.target.value))}
                                className="slider"
                            />
                            <span className="slider-value">{pointSize}</span>
                        </div>
                    </div>

                    <div className="control-row">
                        <label className="toggle-label">
                            <input
                                type="checkbox"
                                checked={showCameras}
                                onChange={(e) => setShowCameras(e.target.checked)}
                                className="toggle"
                            />
                            <span className="toggle-slider" />
                            Show Cameras
                        </label>
                    </div>

                    <div className="control-row">
                        <label className="toggle-label">
                            <input
                                type="checkbox"
                                checked={showAxes}
                                onChange={(e) => setShowAxes(e.target.checked)}
                                className="toggle"
                            />
                            <span className="toggle-slider" />
                            Show Axes
                        </label>
                    </div>

                    <div className="control-row">
                        <label className="toggle-label">
                            <input
                                type="checkbox"
                                checked={showStats}
                                onChange={(e) => setShowStats(e.target.checked)}
                                className="toggle"
                            />
                            <span className="toggle-slider" />
                            Show FPS
                        </label>
                    </div>
                </div>

                <div className="controls-group">
                    <h3 className="controls-title">Background</h3>
                    <div className="color-options">
                        {['#0a0a1a', '#1a1a2e', '#0f172a', '#18181b', '#1e1e1e', '#f5f5f5'].map(c => (
                            <button
                                key={c}
                                className={`color-swatch ${bgColor === c ? 'active' : ''}`}
                                style={{ backgroundColor: c }}
                                onClick={() => setBgColor(c)}
                            />
                        ))}
                    </div>
                </div>

                <div className="controls-group">
                    <h3 className="controls-title">Export</h3>
                    <a
                        href={`/api/result/${sessionId}/pointcloud`}
                        download="reconstruction.ply"
                        className="btn btn-outline btn-sm btn-full"
                    >
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z" />
                            <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z" />
                        </svg>
                        Download PLY
                    </a>
                </div>

                <div className="controls-group controls-help">
                    <p><strong>Navigation:</strong></p>
                    <p>🖱️ Left drag — Orbit</p>
                    <p>🖱️ Right drag — Pan</p>
                    <p>🖱️ Scroll — Zoom</p>
                </div>
            </div>

            {/* 3D Canvas */}
            <div className="viewer-canvas">
                <Canvas
                    camera={{ fov: 60, near: 0.01, far: 10000 }}
                    gl={{ antialias: true, alpha: false }}
                    style={{ background: bgColor }}
                >
                    <color attach="background" args={[bgColor]} />

                    {/* Ambient light */}
                    <ambientLight intensity={0.5} />

                    {/* Point cloud */}
                    <PointCloud positions={positions} colors={colors} pointSize={pointSize} />

                    {/* Camera frustums */}
                    {showCameras && cameraData && cameraData.map((cam, idx) => (
                        <CameraFrustum
                            key={idx}
                            pose={cam.pose}
                            focal={cam.focal}
                            color={`hsl(${(idx / cameraData.length) * 360}, 80%, 65%)`}
                            size={0.15}
                        />
                    ))}

                    {/* Axes helper */}
                    {showAxes && <axesHelper args={[2]} />}

                    {/* Floating background particles */}
                    <FloatingParticles />

                    {/* Scene auto-setup */}
                    <SceneSetup positions={positions} />

                    {/* Controls */}
                    <OrbitControls
                        enableDamping
                        dampingFactor={0.08}
                        rotateSpeed={0.8}
                        zoomSpeed={1.2}
                        panSpeed={0.8}
                    />

                    {/* Stats overlay */}
                    {showStats && <Stats />}
                </Canvas>

                {/* Viewer overlay info */}
                <div className="viewer-overlay">
                    <div className="viewer-badge">
                        <span className="badge-dot" />
                        Interactive 3D View
                    </div>
                </div>
            </div>
        </div>
    )
}
