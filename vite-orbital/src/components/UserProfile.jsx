import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { LogOut, Award, History } from 'lucide-react';

function UserProfile({ currentUser, onLogout }) {
    return (
        <Card className="p-4 mb-4 shadow-sm rounded-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
                <h5>Hello, {currentUser?.username || 'Guest'}!</h5>
                <div className="d-flex gap-2"> {/* Use gap-2 for spacing between buttons */}
                    <Button variant="outline-primary" size="sm" onClick={onShowGameHistory}>
                        <History size={16} className="me-2" />Game History
                    </Button>
                    <Button variant="outline-danger" size="sm" onClick={onLogout}>
                        <LogOut size={16} className="me-2" />Logout
                    </Button>
                </div>
            </div>
            <Card className="p-4 mb-4">
                <h6><Award size={20} className="me-2" />Your Stats:</h6>
                <p className="mb-0">Games Played: {currentUser?.games_played}</p>
                <p className="mb-0">Games Won: {currentUser?.games_won}</p>
                <p>Total Differences Found: {currentUser?.total_differences_found}</p>
            </Card>
        </Card>
    );
}

export default UserProfile;