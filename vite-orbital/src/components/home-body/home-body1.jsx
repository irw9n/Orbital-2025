// // frontend/src/components/home-body/Homebody.jsx
// import React, { useState, useRef, useEffect, useCallback } from 'react';
// import { Container, Row, Col, Form, Button, Card, Spinner, Alert } from 'react-bootstrap';
// import axios from 'axios';
// import { Image as ImageIcon, Edit } from 'lucide-react'; // Importing icons for placeholders

// const BACKEND_URL = 'http://localhost:5000'; // Your Flask backend URL

// function Homebody() {
//   const [selectedFile, setSelectedFile] = useState(null);
//   // Change originalImageUrl to store the object URL for immediate preview
//   const [originalImageUrl, setOriginalImageUrl] = useState('');
//   const [modifiedImageUrl, setModifiedImageUrl] = useState('');
//   const [differences, setDifferences] = useState([]); // This will store the "truth" difference areas
//   const [foundDifferences, setFoundDifferences] = useState(new Set()); // IDs of found differences
//   const [clickAttempts, setClickAttempts] = useState([]); // Stores { x, y, type: 'correct'/'wrong' } for drawing circles
//   const [message, setMessage] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [gameStarted, setGameStarted] = useState(false);

//   // References for the canvas and images to get dimensions
//   const modifiedImageRef = useRef(null);
//   const canvasRef = useRef(null);
//   const fileInputRef = useRef(null); // Ref for the hidden file input

//   const MAX_WRONG_CLICKS = 3; // Limit for wrong clicks

//   // --- START: Moved style declarations here ---
//   const cardBodyStyle = {
//     minHeight: '400px', // Set a minimum height for the card body
//     display: 'flex',
//     justifyContent: 'center',
//     alignItems: 'center',
//     padding: '0', // Remove default padding to allow image to fill
//     overflow: 'hidden' // Hide overflow if image is object-fit: cover
//   };

//   const imageStyle = {
//     width: '90%',
//     height: '90%',
//     objectFit: 'contain', // Scales the image to fit within the content box, preserving aspect ratio.
//   };
//   // --- END: Moved style declarations here ---

//   // Function to draw circles on the canvas
//   const drawCircles = useCallback(() => {
//     const canvas = canvasRef.current;
//     const ctx = canvas.getContext('2d');
//     const img = modifiedImageRef.current;

//     if (!img || !canvas || !ctx) return;

//     // Ensure canvas matches image size
//     canvas.width = img.offsetWidth;
//     canvas.height = img.offsetHeight;

//     ctx.clearRect(0, 0, canvas.width, ctx.height); // Clear previous drawings

//     // Draw circles for correct and incorrect clicks
//     clickAttempts.forEach(attempt => {
//       const { x, y, type } = attempt;
//       ctx.beginPath();
//       ctx.arc(x, y, 20, 0, Math.PI * 2); // Circle with radius 20
//       ctx.lineWidth = 3;
//       if (type === 'correct') {
//         ctx.strokeStyle = 'green';
//       } else if (type === 'wrong') {
//         ctx.strokeStyle = 'red';
//       }
//       ctx.stroke();
//     });

//     // Optionally, draw permanent highlights for found differences after a game ends
//     // or when certain criteria are met.
//     if (foundDifferences.size === differences.length && differences.length > 0) {
//       // Game over, all found, highlight all correctly
//       differences.forEach(diff => {
//         const [x1, y1, x2, y2] = diff.coords;
//         const centerX = (x1 + x2) / 2;
//         const centerY = (y1 + y2) / 2;
//         const radius = Math.max((x2 - x1) / 2, (y2 - y1) / 2, 20); // Dynamic radius

//         ctx.beginPath();
//         ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
//         ctx.lineWidth = 3;
//         ctx.strokeStyle = 'lime'; // Bright green for final reveal
//         ctx.stroke();
//       });
//     }

//   }, [clickAttempts, foundDifferences, differences]); // Redraw when clicks or found differences change

//   // Redraw when images load or window resizes
//   useEffect(() => {
//     const img = modifiedImageRef.current;
//     if (img && img.complete) {
//       drawCircles();
//     }
//   }, [originalImageUrl, modifiedImageUrl, drawCircles]); // Run when images change

//   useEffect(() => {
//     window.addEventListener('resize', drawCircles);
//     return () => window.removeEventListener('resize', drawCircles);
//   }, [drawCircles]);

//   // Effect to clean up the object URL when originalImageUrl changes or component unmounts
//   useEffect(() => {
//     // If originalImageUrl is a temporary object URL, revoke it when it changes
//     // or when the component unmounts. This prevents memory leaks.
//     return () => {
//       if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
//         URL.revokeObjectURL(originalImageUrl);
//       }
//     };
//   }, [originalImageUrl]);


//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     setSelectedFile(file);

//     // Reset game state when a new file is selected
//     setModifiedImageUrl(''); // Clear modified image preview
//     setDifferences([]);
//     setFoundDifferences(new Set());
//     setClickAttempts([]);
//     setMessage('');
//     setError('');
//     setGameStarted(false);

//     // Create a temporary URL for the selected file to display it immediately
//     if (file) {
//       const objectUrl = URL.createObjectURL(file);
//       setOriginalImageUrl(objectUrl);
//     } else {
//       setOriginalImageUrl(''); // Clear image if no file is selected
//     }
//   };

//   const handleUpload = async () => {
//     if (!selectedFile) {
//       setError("Please select an image file first.");
//       return;
//     }

//     setLoading(true);
//     setError('');
//     setMessage('');

//     const formData = new FormData();
//     formData.append('image', selectedFile);

//     try {
//       const response = await axios.post(`${BACKEND_URL}/upload-and-process`, formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });

//       const { originalImageUrl: backendOriginalUrl, modifiedImageUrl, rawDifferencesForFrontendDemo } = response.data;

//       // Note: We are now using the URL from the backend for the original image
//       // This is important because the temporary object URL (blob:) is only valid locally.
//       // The backend URL is a persistent path on your server.
//       setOriginalImageUrl(`${BACKEND_URL}${backendOriginalUrl}`);
//       setModifiedImageUrl(`${BACKEND_URL}${modifiedImageUrl}`);
//       setDifferences(rawDifferencesForFrontendDemo); // Store the truth
//       setGameStarted(true);
//       setMessage("Images loaded! Find the differences.");
//       setClickAttempts([]); // Reset click attempts for new game
//       setFoundDifferences(new Set()); // Reset found differences for new game

//     } catch (err) {
//       console.error("Error uploading or processing image:", err);
//       setError("Failed to upload or process image. Please try again.");
//       setGameStarted(false);
//     } finally {
//       setLoading(false);
//     }
//   };

//   // Handle click on the modified image
//   const handleImageClick = (event) => {
//     if (!gameStarted || foundDifferences.size === differences.length) return; // Don't allow clicks if game not started or finished

//     const img = modifiedImageRef.current;
//     if (!img) return;

//     // Get click coordinates relative to the image (scaled)
//     const rect = img.getBoundingClientRect();
//     const scaleX = img.naturalWidth / rect.width;
//     const scaleY = img.naturalHeight / rect.height;

//     const clickX = (event.clientX - rect.left) * scaleX;
//     const clickY = (event.clientY - rect.top) * scaleY;

//     // Check if the click is within any unfound difference area
//     let isCorrectClick = false;
//     let foundDiffId = null;

//     for (const diff of differences) {
//       if (!foundDifferences.has(diff.id)) { // Only check unfound differences
//         const [x1, y1, x2, y2] = diff.coords;
//         if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
//           isCorrectClick = true;
//           foundDiffId = diff.id;
//           break; // Found a difference, no need to check others
//         }
//       }
//     }

//     // --- Game Logic ---
//     if (isCorrectClick) {
//       if (foundDiffId) {
//         setFoundDifferences(prev => new Set(prev).add(foundDiffId));
//         setClickAttempts(prev => [...prev, {
//             x: (event.clientX - rect.left), // Store scaled for canvas drawing
//             y: (event.clientY - rect.top),
//             type: 'correct'
//         }]);
//         setMessage("Difference found! Keep going!");

//         if (foundDifferences.size + 1 === differences.length) {
//           setMessage("Congratulations! You found all the differences!");
//           setGameStarted(false); // End game
//           drawCircles(); // Ensure all highlights are drawn
//         }
//       }
//     } else {
//       const wrongClicks = clickAttempts.filter(attempt => attempt.type === 'wrong').length;
//       if (wrongClicks < MAX_WRONG_CLICKS) {
//         setClickAttempts(prev => [...prev, {
//             x: (event.clientX - rect.left), // Store scaled for canvas drawing
//             y: (event.clientY - rect.top),
//             type: 'wrong'
//         }]);
//         setMessage(`Oops! Wrong spot. You have ${MAX_WRONG_CLICKS - (wrongClicks + 1)} tries left.`);
//       } else {
//         setMessage(`Game Over! You made too many wrong clicks. The differences are now revealed.`);
//         setGameStarted(false); // End game
//         // Reveal all remaining differences (set them as found to trigger drawing)
//         setFoundDifferences(new Set(differences.map(d => d.id)));
//         drawCircles(); // Explicitly redraw to show all final differences
//       }
//     }
//   };

//   // Function to trigger the hidden file input
//   const triggerFileInput = () => {
//     fileInputRef.current.click();
//   };

//   return (
//     <Container className="my-5">
//       {/* Main Control Card (for file selection and messages) */}

//       <Row className="justify-content-center">
//         {/* Left Image Card: Original Image */}
//         <Col md={6} className="mb-3">
//           <Card className="h-100 shadow-sm">
//             <Form>
//             {/* Hidden file input */}
//             <Form.Control
//                 type="file"
//                 accept="image/*"
//                 onChange={handleFileChange}
//                 ref={fileInputRef}
//                 className="d-none" // Hide the default file input
//             />
//             </Form>
//             <Card.Header className="text-center bg-dark text-white">Original Image</Card.Header>
//             <Card.Body style={cardBodyStyle}>
//               {originalImageUrl ? (
//                 <img src={originalImageUrl} alt="Original" className="img-fluid" style={imageStyle} />
//               ) : (
//                 <div className="text-center text-muted d-flex flex-column align-items-center">
//                   <ImageIcon size={64} className="mb-3" />
//                   <Button variant="link" className="p-0 border-0 text-decoration-none" onClick={triggerFileInput}>
//                     <p className="mb-0">Upload an image to begin</p>
//                   </Button>
//                 </div>
//               )}
//             </Card.Body>
//           </Card>
//         </Col>

//         {/* Right Image Card: Modified Image */}
//         <Col md={6} className="mb-3">
//           <Card className="h-100 shadow-sm">
//             <Card.Header className="text-center bg-primary text-white">Modified Image (Click on the image below!)</Card.Header>
//             <Card.Body style={{ ...cardBodyStyle, position: 'relative' }}>
//               {modifiedImageUrl ? (
//                 <>
//                   <img
//                     ref={modifiedImageRef}
//                     src={modifiedImageUrl}
//                     alt="Modified"
//                     className="img-fluid"
//                     style={{ ...imageStyle, cursor: gameStarted && foundDifferences.size < differences.length ? 'pointer' : 'default' }}
//                     onClick={handleImageClick}
//                     onLoad={drawCircles} // Redraw circles when image loads
//                   />
//                   <canvas
//                     ref={canvasRef}
//                     style={{
//                       position: 'absolute',
//                       top: 0,
//                       left: 0,
//                       width: '100%',
//                       height: '100%',
//                       pointerEvents: 'none', // Allow clicks to pass through to the image
//                     }}
//                   />
//                 </>
//               ) : (
//                 <div className="text-center text-muted d-flex flex-column align-items-center">
//                   <Edit size={64} className="mb-3" />
//                   <p className="mb-0">Modified image will appear here</p>
//                   <Button
//                     variant="success"
//                     onClick={handleUpload}
//                     disabled={!selectedFile || loading} // Disable if no file selected or loading
//                     className="mt-3" // Add some top margin for spacing
//                   >
//                     {loading ? <Spinner animation="border" size="sm" /> : 'Generate Modified Image'}
//                   </Button>
//                 </div>
//               )}
//             </Card.Body>
//           </Card>
//         </Col>
//       </Row>
//     </Container>
//   );
// }

// export default Homebody;