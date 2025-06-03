    import React, { useState, useRef, useEffect, useCallback } from 'react';
    import './home-body.css';
    import { Container, Row, Col, Form, Button, Card, Spinner, Alert } from 'react-bootstrap';
    import axios from 'axios';
    import { Image as ImageIcon, Edit } from 'lucide-react';
    import { v4 as uuidv4 } from 'uuid';

    const BACKEND_URL = 'http://localhost:5000'; // Connecting to Flask backend

    function Homebody() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [originalImageUrl, setOriginalImageUrl] = useState('');
    const [modifiedImageUrl, setModifiedImageUrl] = useState('');
    const [differences, setDifferences] = useState([]);
    const [foundDifferences, setFoundDifferences] = useState(new Set()); 
    const [clickAttempts, setClickAttempts] = useState([]); 
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [gameStarted, setGameStarted] = useState(false);

    // References for the canvas and images to get dimensions
    const modifiedImageRef = useRef(null);
    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const MAX_WRONG_CLICKS = 3; 

    //Style declarations 
    const cardBodyStyle = {
        minHeight: '400px', 
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '0', 
        overflow: 'hidden', 
    };

    const imageStyle = {
        width: '90%', 
        height: '90%',
        objectFit: 'contain', 
    };

    // Function to draw circles on the canvas
    const drawCircles = useCallback(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = modifiedImageRef.current;

        if (!img || !canvas || !ctx) return;

        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;

        const displayedWidth = img.offsetWidth;
        const displayedHeight = img.offsetHeight;

        canvas.width = displayedWidth;
        canvas.height = displayedHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height); 

        // Draw circles for correct and incorrect clicks
        clickAttempts.forEach(attempt => {
        const { x, y, type } = attempt; 
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, Math.PI * 2); 
        ctx.lineWidth = 3;
        if (type === 'correct') {
            ctx.strokeStyle = 'green';
        } else if (type === 'wrong') {
            ctx.strokeStyle = 'red';
        }
        ctx.stroke();
        });

        // If game is over (all found or too many wrong clicks), reveal all differences
        if (foundDifferences.size === differences.length && differences.length > 0) {
        differences.forEach(diff => {
            const [x1_natural, y1_natural, x2_natural, y2_natural] = diff.coords;

            // Scale natural coordinates to displayed coordinates for drawing
            const scaleX = displayedWidth / naturalWidth;
            const scaleY = displayedHeight / naturalHeight;

            const x1_display = x1_natural * scaleX;
            const y1_display = y1_natural * scaleY;
            const x2_display = x2_natural * scaleX;
            const y2_display = y2_natural * scaleY;

            const centerX_display = (x1_display + x2_display) / 2;
            const centerY_display = (y1_display + y2_display) / 2;

            const radius = Math.max((x2_display - x1_display) / 2, (y2_display - y1_display) / 2, 20);

            ctx.beginPath();
            ctx.arc(centerX_display, centerY_display, radius, 0, Math.PI * 2);
            ctx.lineWidth = 3;
            ctx.strokeStyle = 'lime'; 
            ctx.stroke();
        });
        }

    }, [clickAttempts, foundDifferences, differences]); // Redraw when dependencies change

    // Redraw when modifiedImageUrl changes or window resizes
    useEffect(() => {
        const img = modifiedImageRef.current;
        if (img && img.complete) { // Ensure image is fully loaded before drawing
        drawCircles();
        }
    }, [modifiedImageUrl, drawCircles]); // Run when modified image URL changes

    useEffect(() => {
        window.addEventListener('resize', drawCircles);
        // Add an event listener to the image itself in case it loads AFTER component mounts
        const img = modifiedImageRef.current;
        if (img) {
        img.addEventListener('load', drawCircles);
        }

        return () => {
        window.removeEventListener('resize', drawCircles);
        if (img) {
            img.removeEventListener('load', drawCircles);
        }
        };
    }, [drawCircles]); 

    useEffect(() => {
        return () => {
        if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
            URL.revokeObjectURL(originalImageUrl);
        }
        };
    }, [originalImageUrl]);


    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);

        // Reset game state when a new file is selected
        setModifiedImageUrl(''); 
        setDifferences([]);
        setFoundDifferences(new Set());
        setClickAttempts([]);
        setMessage('');
        setError('');
        setGameStarted(false);

        // Create a temporary URL for the selected file to display it immediately
        if (file) {
        const objectUrl = URL.createObjectURL(file);
        setOriginalImageUrl(objectUrl);
        } else {
        setOriginalImageUrl('');
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
        setError("Please select an image file first.");
        return;
        }

        setLoading(true);
        setError('');
        setMessage('');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
        const response = await axios.post(`${BACKEND_URL}/upload-and-process`, formData, {
            headers: {
            'Content-Type': 'multipart/form-data',
            },
        });

        const { originalImageUrl: backendOriginalUrl, modifiedImageUrl, rawDifferencesForFrontendDemo } = response.data;

        // Revoke the temporary blob URL for the original image if it exists
        if (originalImageUrl.startsWith('blob:')) {
            URL.revokeObjectURL(originalImageUrl);
        }

        setOriginalImageUrl(`${BACKEND_URL}${backendOriginalUrl}`);
        setModifiedImageUrl(`${BACKEND_URL}${modifiedImageUrl}`);

        // Assign a unique ID to each difference for tracking found differences
        const differencesWithIds = rawDifferencesForFrontendDemo.map(coords => ({
            id: uuidv4(), // Generate a unique ID for each difference
            coords: coords, 
        }));
        setDifferences(differencesWithIds);

        setGameStarted(true);
        setMessage("Images loaded! Find the differences.");
        setClickAttempts([]); // Reset click attempts for new game
        setFoundDifferences(new Set()); // Reset found differences for new game

        } catch (err) {
        console.error("Error uploading or processing image:", err);
        if (err.response && err.response.data && err.response.data.error) {
            setError(`Failed to upload or process image: ${err.response.data.error}`);
        } else {
            setError("Failed to upload or process image. Please try again.");
        }
        setGameStarted(false);
        } finally {
        setLoading(false);
        }
    };

    // Handle click on the modified image
    const handleImageClick = (event) => {
        if (!gameStarted || foundDifferences.size === differences.length || clickAttempts.filter(a => a.type === 'wrong').length >= MAX_WRONG_CLICKS) {
        // Don't allow clicks if game not started, finished, or too many wrong clicks
        return;
        }

        const img = modifiedImageRef.current;
        if (!img) return;

        // Get click coordinates relative to the image element
        const rect = img.getBoundingClientRect();
        const clickX_display = event.clientX - rect.left;
        const clickY_display = event.clientY - rect.top;

        // Scale click coordinates to the natural (backend) dimensions of the image
        const scaleX = img.naturalWidth / img.offsetWidth;
        const scaleY = img.naturalHeight / img.offsetHeight;

        const clickX_natural = clickX_display * scaleX;
        const clickY_natural = clickY_display * scaleY;

        // Check if the click is within any unfound difference area
        let isCorrectClick = false;
        let foundDiffId = null;

        for (const diff of differences) {
        if (!foundDifferences.has(diff.id)) { 
            const [x1, y1, x2, y2] = diff.coords; 

            const tolerance = 10; 
            if (
            clickX_natural >= (x1 - tolerance) &&
            clickX_natural <= (x2 + tolerance) &&
            clickY_natural >= (y1 - tolerance) &&
            clickY_natural <= (y2 + tolerance)
            ) {
            isCorrectClick = true;
            foundDiffId = diff.id;
            break; // Found a difference, no need to check others
            }
        }
        }

        // --- Game Logic ---
        if (isCorrectClick) {
        if (foundDiffId) {
            setFoundDifferences(prev => new Set(prev).add(foundDiffId));
            setClickAttempts(prev => [...prev, {
                x: clickX_display, 
                y: clickY_display,
                type: 'correct'
            }]);
            setMessage("Difference found! Keep going!");

            // Check if all differences are found
            if (foundDifferences.size + 1 === differences.length) { 
            setMessage("Congratulations! You found all the differences!");
            setGameStarted(false); // End game

            setTimeout(() => drawCircles(), 50); 
            }
        }
        } else {
        const wrongClicks = clickAttempts.filter(attempt => attempt.type === 'wrong').length;
        if (wrongClicks < MAX_WRONG_CLICKS - 1) { 
            setClickAttempts(prev => [...prev, {
                x: clickX_display, 
                y: clickY_display,
                type: 'wrong'
            }]);
            setMessage(`Oops! Wrong spot. You have ${MAX_WRONG_CLICKS - (wrongClicks + 1)} tries left.`);
        } else { // This is the last wrong click
            setClickAttempts(prev => [...prev, {
                x: clickX_display,
                y: clickY_display,
                type: 'wrong'
            }]);
            setMessage(`Game Over! You made too many wrong clicks. The differences are now revealed.`);
            setGameStarted(false); // End game
            
            setFoundDifferences(new Set(differences.map(d => d.id))); // Reveal all differences
            
            setTimeout(() => drawCircles(), 50); // Trigger redraw to show all highlights immediately
        }
        }
    };

    // Function to trigger the hidden file input
    const triggerFileInput = () => {
        fileInputRef.current.click();
    };

    return (
        <Container className="my-5">
        {/* Main Control Card (for file selection and messages) */}
        <Row className="mb-3 justify-content-center">
            <Col md={12}>
                {error && <Alert variant="danger">{error}</Alert>}
                {message && <Alert variant="info">{message}</Alert>}
            </Col>
        </Row>

        <Row className="justify-content-center">
            {/* Left Image Card: Original Image */}
            <Col md={6} className="mb-3">
            <Card className="h-100 shadow-sm">
                <Form>
                    {/* Hidden file input */}
                    <Form.Control
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        ref={fileInputRef}
                        className="d-none" 
                    />
                </Form>
                <Card.Header className="text-center bg-dark text-white">Original Image</Card.Header>
                <Card.Body style={cardBodyStyle}>
                {originalImageUrl ? (
                    <img key={originalImageUrl} src={originalImageUrl} alt="Original" className="img-fluid" style={imageStyle} />
                ) : (
                    <div className="text-center text-muted d-flex flex-column align-items-center">
                    <ImageIcon size={64} className="mb-3" />
                    <Button variant="link" className="p-0 border-0 text-decoration-none" onClick={triggerFileInput}>
                        <p className="mb-0">Upload an image to begin</p>
                    </Button>
                    </div>
                )}
                </Card.Body>
            </Card>
            </Col>

            {/* Right Image Card: Modified Image */}
            <Col md={6} className="mb-3">
            <Card className="h-100 shadow-sm">
                <Card.Header className="text-center bg-primary text-white">Modified Image (Click on the image below!)</Card.Header>
                <Card.Body style={{ ...cardBodyStyle, position: 'relative' }}>
                {modifiedImageUrl ? (
                    <>
                    <img
                        ref={modifiedImageRef}
                        key={modifiedImageUrl}
                        src={modifiedImageUrl}
                        alt="Modified"
                        className="img-fluid"
                        style={{ ...imageStyle, cursor: gameStarted && foundDifferences.size < differences.length && clickAttempts.filter(a => a.type === 'wrong').length < MAX_WRONG_CLICKS ? 'pointer' : 'default' }}
                        onClick={handleImageClick}
                    />
                    <canvas
                        ref={canvasRef}
                        style={{
                        position: 'absolute',
                        top: '5%',
                        left: '5%', 
                        width: '90%',
                        height: '90%', 
                        pointerEvents: 'none',
                        }}
                    />
                    </>
                ) : (
                    <div className="text-center text-muted d-flex flex-column align-items-center">
                    <Edit size={64} className="mb-3" />
                    <p className="mb-0">Modified image will appear here</p>
                    <Button
                        variant="success"
                        onClick={handleUpload}
                        disabled={!selectedFile || loading} 
                        className="mt-3"
                    >
                        {loading ? <Spinner animation="border" size="sm" /> : 'Generate Modified Image'}
                    </Button>
                    </div>
                )}
                </Card.Body>
            </Card>
            </Col>
        </Row>
        </Container>
    );
    }

    export default Homebody;