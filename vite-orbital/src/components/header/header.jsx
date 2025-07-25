import React from 'react';
import './header.css';
import {Container, Nav, Navbar, NavDropdown} from 'react-bootstrap';
import { LogIn, UserPlus, User, LogOut, Award } from 'lucide-react';

const Header = ({ isLoggedIn, currentUser, onLoginClick, onRegisterClick, onLogout }) => {
    return (
    <Navbar expand="lg" className="bg-body-tertiary">
      <Container>
        <Navbar.Brand href="#home">spotthedifference</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link href="#home">How To Play</Nav.Link>
            <NavDropdown title="Account" id="basic-nav-dropdown">
              {!isLoggedIn ? (
                <>
                  <NavDropdown.Item onClick={onLoginClick}>
                      <LogIn size={16} className="me-2" />Login
                  </NavDropdown.Item>
                  <NavDropdown.Item onClick={onRegisterClick}>
                      <UserPlus size={16} className="me-2" />Register
                  </NavDropdown.Item>
                </>
              ) : (
                <>
                  <NavDropdown.Item disabled>
                      <User size={16} className="me-2" />Logged in as: {currentUser?.username}
                  </NavDropdown.Item>
                  <NavDropdown.Divider />
                  <NavDropdown.Item href="#my-stats">
                      <Award size={16} className="me-2" />My Profile
                  </NavDropdown.Item>
                  <NavDropdown.Item onClick={onLogout}>
                      <LogOut size={16} className="me-2" />Logout
                  </NavDropdown.Item>
                </>
              )}
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
    );
    }

export default Header;
