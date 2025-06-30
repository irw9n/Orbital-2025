    import React, { useState, useRef, useEffect, useCallback } from 'react';
    import './home-body.css';
    import { Container, Row, Col, Form, Button, Card, Spinner, Alert, Tabs, Tab } from 'react-bootstrap';
    import axios from 'axios';
    import { Image as ImageIcon, Edit, User as UserIcon, LogIn, LogOut, Award } from 'lucide-react';
    import { v4 as uuidv4 } from 'uuid';

    // const BACKEND_URL = 'http://localhost:5000'; // Connecting to Local Flask backend
    // const BACKEND_URL = 'https://orbital-2025-backend.onrender.com'; // Connecting to Local Flask backend
    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'https://orbital-2025-backend.onrender.com'; //for vercel deployment


    axios.defaults.withCredentials = true; // Enable sending cookies with requests

    function Homebody() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [authMessage, setAuthMessage] = useState('');
    const [authError, setAuthError] = useState('');
    const [activeTab, setActiveTab] = useState('login');

    const [regUsername, setRegUsername] = useState('');
    const [regEmail, setRegEmail] = useState('');
    const [regPassword, setRegPassword] = useState('');

    const [logUsername, setLogUsername] = useState('');
    const [logPassword, setLogPassword] = useState('');

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

    useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        // This endpoint will return user profile if logged in, or 401 if not
        const response = await axios.get(`${BACKEND_URL}/user_profile`);
        setIsLoggedIn(true);
        setCurrentUser(response.data);
        setAuthMessage(`Welcome back, ${response.data.username}!`);
      } catch (err) {
        if (err.reponse && err.response.status === 401) {
            setIsLoggedIn(false);
            setCurrentUser(null);
            setAuthMessage('Please log in or register to play.');
            setAuthError(''); // Clear any previous auth errors on initial check
        }
        else {
            console.error("Error checking login status:", err);
            setIsLoggedIn(false);
            setCurrentUser(null);
            setAuthMessage('An error occurred. Please try again later.');
            setAuthError(err.response?.data?.error || "Failed to check login status.");
        }
      }
    };
    checkLoginStatus();}, []);

    const handleRegister = async (event) => {
        event.preventDefault();
        setAuthMessage('');
        setAuthError('');
        

        if (!regUsername  || !regEmail || !regPassword) {
            setAuthError('All fields are required for registration.');
            return;
        }

        try {
            const response = await axios.post(`${BACKEND_URL}/register`, { 
                username: regUsername, 
                email: regEmail, 
                password: regPassword
            });
            setAuthMessage(response.data.message + " You can now log in.");
            setActiveTab('login');
            setRegUsername('');
            setRegEmail('');
            setRegPassword('');
        }
        catch (err) {
            console.error("Registration error:", err);
            setAuthError('Registration error. Please try again.');
        }
    };

    const handleLogin = async (event) => {
        event.preventDefault();
        setAuthMessage('');
        setAuthError('');
        

        if (!logUsername || !logPassword) {
            setAuthError('Username and password are required for login.');
            return;
        }

        try {
            const response = await axios.post(`${BACKEND_URL}/login`, { 
                username: logUsername, 
                password: logPassword  
            });

            setIsLoggedIn(true);
            setCurrentUser(response.data);
            setAuthMessage(response.data.message);

            setLogUsername('');
            setLogPassword('');

            setSelectedFile(null);
            setOriginalImageUrl('');
            setModifiedImageUrl('');
            setDifferences([]);
            setFoundDifferences(new Set());
            setClickAttempts([]);
            setMessage('');
            setError('');
            setGameStarted(false);
        }
        catch (err) {
            console.error("Login error:", err);
            setAuthError("Login failed. Please check your credentials.");
        }
    };

    const handleLogout = async () => {
        try {
            await axios.post(`${BACKEND_URL}/logout`);
            setIsLoggedIn(false);
            setCurrentUser(null);
            setAuthMessage("You have been logged out.");
            setAuthError('');
            setSelectedFile(null);
            setOriginalImageUrl('');
            setModifiedImageUrl('');
            setDifferences([]);
            setFoundDifferences(new Set());
            setClickAttempts([]);
            setMessage('');
            setError('');
            setGameStarted(false);
        }
        catch (err) {
            console.error("Logout error:", err);
            setAuthError("Logout failed.");
        }
    };

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

        if (!isLoggedIn) {
            setError("Please login to upload an image.");
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

        const profileResponse = await axios.get(`${BACKEND_URL}/user_profile`);
        setCurrentUser(profileResponse.data); // Update current user profile after upload
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
    const handleImageClick = async (event) => {
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

            const tolerance = 15; 
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
            const updatedFoundDifferencesSet = new Set(foundDifferences);
            updatedFoundDifferencesSet.add(foundDiffId);

            setFoundDifferences(updatedFoundDifferencesSet);

            const currentFoundCount = updatedFoundDifferencesSet.size;

            setClickAttempts(prev => [...prev, {
                x: clickX_display, 
                y: clickY_display,
                type: 'correct'
            }]);
            setMessage("Difference found! Keep going!");

            // Check if all differences are found
            if (currentFoundCount === differences.length) { 
            setMessage("Congratulations! You found all the differences!");
            setGameStarted(false); // End game

            setTimeout(() => drawCircles(), 50); 

            try {
                await axios.post(`${BACKEND_URL}/update_stats`, {
                    differencesFound: updatedFoundDifferencesSet.size,
                    gameWon: true
                });

                const profileResponse = await axios.get(`${BACKEND_URL}/user_profile`);
                setCurrentUser(profileResponse.data);
                setAuthMessage(prev => prev + " Your stats have been updated!");
            } catch (err) {
                console.error("Failed to update stats:", err);
                setError("Failed to update game statistics.");
            }
            
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
            
            try {
                await axios.post(`${BACKEND_URL}/update_stats`, {
                    differencesFound: foundDifferences.size, // Only count differences actually found
                    gameWon: false
                });

                const profileResponse = await axios.get(`${BACKEND_URL}/user_profile`);
                setCurrentUser(profileResponse.data);
                setAuthMessage(prev => prev + " Your stats have been updated!");
            } catch (err) {
                console.error("Failed to update stats:", err);
                setError("Failed to update game statistics.");
            }
        }
        }
    };

    // Function to trigger the hidden file input
    const triggerFileInput = () => {
        fileInputRef.current.click();
    };

    return (
        <Container className="my-5">

        {/* Authentication and Messages Card */}
        <Card className="p-4 shadow-sm mb-4">
            {authError && <Alert variant="danger" className="mt-3">{authError}</Alert>}
            {authMessage && <Alert variant="info" className="mt-3">{authMessage}</Alert>}

            {!isLoggedIn ? (
            // Login/Register Tabs if not logged in
            <Tabs activeKey={activeTab} onSelect={(k) => setActiveTab(k)} className="mb-3">
                <Tab eventKey="login" title={<span><LogIn size={16} className="me-2" />Login</span>}>
                <Form onSubmit={handleLogin} className="mt-3">
                    <Form.Group className="mb-3" controlId="loginUsername">
                    <Form.Label>Username</Form.Label>
                    <Form.Control type="text" placeholder="Enter username" required value={logUsername}
                    onChange={(e) => setLogUsername(e.target.value)}/>
                    </Form.Group>
                    <Form.Group className="mb-3" controlId="loginPassword">
                    <Form.Label>Password</Form.Label>
                    <Form.Control type="password" placeholder="Password" required value={logPassword}
                    onChange={(e) => setLogPassword(e.target.value)}/>
                    </Form.Group>
                    <Button variant="primary" type="submit">Login</Button>
                </Form>
                </Tab>
                <Tab eventKey="register" title={<span><UserIcon size={16} className="me-2" />Register</span>}>
                <Form onSubmit={handleRegister} className="mt-3">
                    <Form.Group className="mb-3" controlId="registerUsername">
                    <Form.Label>Username</Form.Label>
                    <Form.Control type="text" placeholder="Choose a username" required value={regUsername} 
                    onChange={(e) => setRegUsername(e.target.value)}/>
                    </Form.Group>
                    <Form.Group className="mb-3" controlId="registerEmail">
                    <Form.Label>Email address</Form.Label>
                    <Form.Control type="email" placeholder="Enter email" required value={regEmail}
                    onChange={(e) => setRegEmail(e.target.value)}/>
                    </Form.Group>
                    <Form.Group className="mb-3" controlId="registerPassword">
                    <Form.Label>Password</Form.Label>
                    <Form.Control type="password" placeholder="Password" required value={regPassword}
                    onChange={(e) => setRegPassword(e.target.value)}/>
                    <Form.Text className="text-muted">
                        Your password must be at least 6 characters long.
                    </Form.Text>
                    </Form.Group>
                    <Button variant="primary" type="submit">Register</Button>
                </Form>
                </Tab>
            </Tabs>
            ) : (
            // User profile and file upload UI if logged in
            <>
                <div className="d-flex justify-content-between align-items-center mb-3">
                <h5>Hello, {currentUser?.username || 'Guest'}!</h5>
                <Button variant="outline-danger" size="sm" onClick={handleLogout}>
                    <LogOut size={16} className="me-2" />Logout
                </Button>
                </div>
                <Card className="p-4 mb-4">
                <h6><Award size={20} className="me-2" />Your Stats:</h6>
                <p className="mb-0">Games Played: {currentUser?.games_played}</p>
                <p className="mb-0">Games Won: {currentUser?.games_won}</p>
                <p>Total Differences Found: {currentUser?.total_differences_found}</p>
                <hr/> {/* Separator */}
                {/* File input and upload button moved here */}
                <Form>
                    {/* Hidden file input */}
                    <Form.Control
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    className="d-none" // Hide the default file input
                    />
                    {/* Visible File Selection UI */}
                    <Form.Group controlId="formFile" className="mb-3 d-flex align-items-center">
                    <Form.Label className="mb-0 me-3">Select your image:</Form.Label>
                    <Button variant="primary" onClick={triggerFileInput} disabled={loading} className="me-2">
                        Browse Image
                    </Button>
                    {selectedFile && <span className="ms-3 text-muted text-truncate" style={{ maxWidth: '200px' }}>Selected: {selectedFile.name}</span>}
                    </Form.Group>
                </Form>
                </Card>
            </>
            )}
        </Card>
        

        {/* Main Control Card (for file selection and messages) (only shown if logged in) */}
        { isLoggedIn && (
        <>
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
        </>
        )}
        </Container>
    );
    }

    export default Homebody;