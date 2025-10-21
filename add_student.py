#!/usr/bin/env python3
"""
Quick script to add students to the database
Usage: python add_student.py
"""

from database import add_student, get_all_students, init_db

def main():
    print("=" * 50)
    print("Add Student to Behavior Analysis System")
    print("=" * 50)
    print()
    
    # Initialize database if needed
    init_db()
    
    # Get student info
    student_id = input("Enter Student ID (e.g., S001): ").strip()
    if not student_id:
        print("❌ Student ID is required!")
        return
    
    name = input("Enter Student Name: ").strip()
    if not name:
        print("❌ Name is required!")
        return
    
    email = input("Enter Email (optional, press Enter to skip): ").strip()
    if not email:
        email = None
    
    # Add student
    success = add_student(student_id, name, email)
    
    if success:
        print()
        print("✅ Student added successfully!")
        print(f"   ID: {student_id}")
        print(f"   Name: {name}")
        if email:
            print(f"   Email: {email}")
    else:
        print()
        print("❌ Failed to add student. Student ID may already exist.")
    
    print()
    print("-" * 50)
    print("All Students in Database:")
    print("-" * 50)
    
    students = get_all_students()
    for s in students:
        email_str = f" ({s['email']})" if s['email'] else ""
        print(f"  • {s['student_id']} - {s['name']}{email_str}")
    
    print()

if __name__ == '__main__':
    main()


