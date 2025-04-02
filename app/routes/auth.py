"""
ProjectPilot - AI-powered project management system
Authentication routes (login, register, logout).
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.urls import url_parse
from app import db
from app.models.user import User

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if current_user.is_authenticated:
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False) == 'true'
        
        user = User.query.filter_by(username=username).first()
        
        if user is None or not user.verify_password(password):
            flash('Invalid username or password.', 'danger')
            return render_template('auth/login.html')
        
        if not user.is_active:
            flash('This account is inactive. Please contact an administrator.', 'warning')
            return render_template('auth/login.html')
        
        login_user(user, remember=remember)
        user.update_last_login()
        
        # Handle next page
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('projects.index')
            
        flash(f'Welcome back, {user.username}!', 'success')
        return redirect(next_page)
        
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Basic validation
        if not all([username, email, password, password_confirm]):
            flash('All fields are required.', 'danger')
            return render_template('auth/register.html')
        
        if password != password_confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('auth/register.html')
            
        # Check for existing user
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return render_template('auth/register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return render_template('auth/register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.password = password  # This sets password_hash via the setter
        
        # Assign admin role to first user
        if User.query.count() == 0:
            user.is_admin = True
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html')

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile route."""
    return render_template('auth/profile.html', user=current_user)

@auth_bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    """Update user profile."""
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('email')
    
    if email and email != current_user.email:
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('auth.profile'))
        current_user.email = email
    
    current_user.first_name = first_name
    current_user.last_name = last_name
    
    db.session.commit()
    flash('Profile updated successfully.', 'success')
    return redirect(url_for('auth.profile'))

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not current_user.verify_password(current_password):
        flash('Current password is incorrect.', 'danger')
        return redirect(url_for('auth.profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('auth.profile'))
    
    current_user.password = new_password
    db.session.commit()
    
    flash('Password changed successfully.', 'success')
    return redirect(url_for('auth.profile'))