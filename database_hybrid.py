"""
Hybrid Database System - AWS DynamoDB + MongoDB
- DynamoDB: Fast access for students, authentication, session data
- MongoDB: Complex queries for reports, frame analysis, audit logs
"""

import os
import boto3
from boto3.dynamodb.conditions import Key, Attr
from pymongo import MongoClient, DESCENDING
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============= AWS DynamoDB Configuration =============

# Initialize DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# DynamoDB Tables
STUDENTS_TABLE = os.getenv('DYNAMODB_STUDENTS_TABLE', 'students')
SESSIONS_TABLE = os.getenv('DYNAMODB_SESSIONS_TABLE', 'sessions')

# ============= MongoDB Configuration =============

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB = os.getenv('MONGODB_DB', 'behavior_analysis')

mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client[MONGODB_DB]

# MongoDB Collections
reports_collection = mongo_db['reports']
frame_analysis_collection = mongo_db['frame_analysis']
audit_log_collection = mongo_db['audit_log']

# Create indexes for performance
reports_collection.create_index([('student_id', 1), ('session_date', DESCENDING)])
frame_analysis_collection.create_index([('report_id', 1), ('frame_number', 1)])
audit_log_collection.create_index([('timestamp', DESCENDING), ('user_id', 1)])


# ============= DynamoDB Functions (Students & Auth) =============

def init_dynamodb_tables():
    """Create DynamoDB tables if they don't exist"""
    try:
        # Students Table
        try:
            students_table = dynamodb.create_table(
                TableName=STUDENTS_TABLE,
                KeySchema=[
                    {'AttributeName': 'student_id', 'KeyType': 'HASH'}  # Partition key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'student_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'  # On-demand pricing
            )
            students_table.wait_until_exists()
            print(f"✓ Created DynamoDB table: {STUDENTS_TABLE}")
        except dynamodb.meta.client.exceptions.ResourceInUseException:
            print(f"✓ DynamoDB table already exists: {STUDENTS_TABLE}")
        
        # Sessions Table (for active sessions)
        try:
            sessions_table = dynamodb.create_table(
                TableName=SESSIONS_TABLE,
                KeySchema=[
                    {'AttributeName': 'session_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'session_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST',
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl'  # Auto-delete expired sessions
                }
            )
            sessions_table.wait_until_exists()
            print(f"✓ Created DynamoDB table: {SESSIONS_TABLE}")
        except dynamodb.meta.client.exceptions.ResourceInUseException:
            print(f"✓ DynamoDB table already exists: {SESSIONS_TABLE}")
            
    except Exception as e:
        print(f"⚠ Error creating DynamoDB tables: {e}")


def add_student(student_id, name, email=None, password=None):
    """
    Add student to DynamoDB
    Fast key-value access for authentication
    """
    try:
        table = dynamodb.Table(STUDENTS_TABLE)
        
        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        item = {
            'student_id': student_id,
            'name': name,
            'email': email or '',
            'password_hash': password_hash or '',
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Conditional put - fail if student already exists
        table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(student_id)'
        )
        
        print(f"✓ Added student to DynamoDB: {student_id}")
        return True
        
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        print(f"⚠ Student already exists: {student_id}")
        return False
    except Exception as e:
        print(f"⚠ Error adding student: {e}")
        return False


def get_student(student_id):
    """
    Get student from DynamoDB
    O(1) access time with partition key
    """
    try:
        table = dynamodb.Table(STUDENTS_TABLE)
        response = table.get_item(Key={'student_id': student_id})
        
        if 'Item' in response:
            item = response['Item']
            return {
                'student_id': item['student_id'],
                'name': item['name'],
                'email': item.get('email', ''),
                'created_at': item.get('created_at', '')
            }
        
        return None
        
    except Exception as e:
        print(f"⚠ Error getting student: {e}")
        return None


def verify_student_password(student_id, password):
    """
    Verify student password from DynamoDB
    Fast authentication check
    """
    try:
        table = dynamodb.Table(STUDENTS_TABLE)
        response = table.get_item(Key={'student_id': student_id})
        
        if 'Item' not in response:
            return False
        
        stored_hash = response['Item'].get('password_hash', '')
        if not stored_hash:
            return False
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == stored_hash
        
    except Exception as e:
        print(f"⚠ Error verifying password: {e}")
        return False


def get_all_students():
    """
    Scan all students from DynamoDB
    Use sparingly - expensive operation
    """
    try:
        table = dynamodb.Table(STUDENTS_TABLE)
        response = table.scan()
        
        students = []
        for item in response.get('Items', []):
            students.append({
                'student_id': item['student_id'],
                'name': item['name'],
                'email': item.get('email', ''),
                'created_at': item.get('created_at', '')
            })
        
        return students
        
    except Exception as e:
        print(f"⚠ Error getting all students: {e}")
        return []


# ============= MongoDB Functions (Reports & Analytics) =============

def save_report(student_id, duration, behavior_stats, frame_data, notes=None):
    """
    Save report to MongoDB
    Better for complex nested data and queries
    """
    try:
        # Calculate totals
        total_frames = sum(stat['count'] for stat in behavior_stats)
        
        # Convert list to dict for easier access
        behavior_counts = {stat['label']: stat['count'] for stat in behavior_stats}
        behavior_percents = {stat['label']: stat['percentage'] for stat in behavior_stats}
        
        # Calculate engagement score
        engagement_score = (
            behavior_percents.get('Raising Hand', 0) * 1.0 +
            behavior_percents.get('Writing', 0) * 0.9 +
            behavior_percents.get('Reading', 0) * 0.8 +
            behavior_percents.get('Sleeping', 0) * 0.1
        ) / 100
        
        # Create report document
        report = {
            'student_id': student_id,
            'session_date': datetime.utcnow(),
            'duration': duration,
            'total_frames': total_frames,
            'behaviors': {
                'raising_hand': {
                    'count': behavior_counts.get('Raising Hand', 0),
                    'percent': behavior_percents.get('Raising Hand', 0)
                },
                'reading': {
                    'count': behavior_counts.get('Reading', 0),
                    'percent': behavior_percents.get('Reading', 0)
                },
                'sleeping': {
                    'count': behavior_counts.get('Sleeping', 0),
                    'percent': behavior_percents.get('Sleeping', 0)
                },
                'writing': {
                    'count': behavior_counts.get('Writing', 0),
                    'percent': behavior_percents.get('Writing', 0)
                }
            },
            'engagement_score': engagement_score,
            'notes': notes or ''
        }
        
        # Insert report
        result = reports_collection.insert_one(report)
        report_id = str(result.inserted_id)
        
        # Insert frame analysis data
        if frame_data:
            frame_docs = []
            for frame in frame_data:
                frame_docs.append({
                    'report_id': report_id,
                    'frame_number': frame.get('frame', 0),
                    'timestamp': frame.get('timestamp', 0),
                    'behavior': frame.get('prediction', ''),
                    'confidence': frame.get('confidence', 0)
                })
            
            if frame_docs:
                frame_analysis_collection.insert_many(frame_docs)
        
        print(f"✓ Saved report to MongoDB: {report_id}")
        return report_id
        
    except Exception as e:
        print(f"⚠ Error saving report: {e}")
        return None


def get_student_reports(student_id):
    """
    Get all reports for a student from MongoDB
    Efficient with indexed queries
    """
    try:
        reports = reports_collection.find(
            {'student_id': student_id}
        ).sort('session_date', DESCENDING)
        
        result = []
        for report in reports:
            result.append({
                'id': str(report['_id']),
                'student_id': report['student_id'],
                'session_date': report['session_date'].isoformat(),
                'duration': report.get('duration', 0),
                'total_frames': report.get('total_frames', 0),
                'behaviors': report.get('behaviors', {}),
                'engagement_score': report.get('engagement_score', 0),
                'notes': report.get('notes', '')
            })
        
        return result
        
    except Exception as e:
        print(f"⚠ Error getting student reports: {e}")
        return []


def get_report_details(report_id):
    """
    Get detailed report with frame analysis from MongoDB
    """
    try:
        from bson.objectid import ObjectId
        
        # Get report
        report = reports_collection.find_one({'_id': ObjectId(report_id)})
        
        if not report:
            return None
        
        # Get frame analysis
        frames = frame_analysis_collection.find(
            {'report_id': report_id}
        ).sort('frame_number', 1)
        
        frame_list = []
        for frame in frames:
            frame_list.append({
                'frame': frame.get('frame_number', 0),
                'timestamp': frame.get('timestamp', 0),
                'behavior': frame.get('behavior', ''),
                'confidence': frame.get('confidence', 0)
            })
        
        return {
            'id': str(report['_id']),
            'student_id': report['student_id'],
            'session_date': report['session_date'].isoformat(),
            'duration': report.get('duration', 0),
            'total_frames': report.get('total_frames', 0),
            'behaviors': report.get('behaviors', {}),
            'engagement_score': report.get('engagement_score', 0),
            'notes': report.get('notes', ''),
            'frame_analysis': frame_list
        }
        
    except Exception as e:
        print(f"⚠ Error getting report details: {e}")
        return None


def add_audit_log(user_id, user_type, action, entity_type=None, entity_id=None, details=None):
    """
    Add audit log to MongoDB
    Time-series data perfect for MongoDB
    """
    try:
        log_entry = {
            'user_id': user_id,
            'user_type': user_type,
            'action': action,
            'entity_type': entity_type or '',
            'entity_id': entity_id or '',
            'details': details or '',
            'timestamp': datetime.utcnow()
        }
        
        audit_log_collection.insert_one(log_entry)
        
    except Exception as e:
        print(f"⚠ Error adding audit log: {e}")


def get_audit_logs(limit=100):
    """
    Get recent audit logs from MongoDB
    Efficient with indexed timestamp
    """
    try:
        logs = audit_log_collection.find().sort('timestamp', DESCENDING).limit(limit)
        
        result = []
        for log in logs:
            result.append({
                'id': str(log['_id']),
                'user_id': log.get('user_id', ''),
                'user_type': log.get('user_type', ''),
                'action': log.get('action', ''),
                'entity_type': log.get('entity_type', ''),
                'entity_id': log.get('entity_id', ''),
                'details': log.get('details', ''),
                'timestamp': log['timestamp'].isoformat()
            })
        
        return result
        
    except Exception as e:
        print(f"⚠ Error getting audit logs: {e}")
        return []


# ============= Advanced MongoDB Queries =============

def get_student_analytics(student_id):
    """
    Get advanced analytics for a student using MongoDB aggregation
    """
    try:
        pipeline = [
            {'$match': {'student_id': student_id}},
            {'$group': {
                '_id': '$student_id',
                'total_sessions': {'$sum': 1},
                'avg_engagement': {'$avg': '$engagement_score'},
                'total_duration': {'$sum': '$duration'},
                'avg_duration': {'$avg': '$duration'}
            }}
        ]
        
        result = list(reports_collection.aggregate(pipeline))
        
        if result:
            return result[0]
        
        return None
        
    except Exception as e:
        print(f"⚠ Error getting analytics: {e}")
        return None


def get_behavior_trends(student_id, behavior_type):
    """
    Get behavior trends over time for a student
    """
    try:
        pipeline = [
            {'$match': {'student_id': student_id}},
            {'$sort': {'session_date': 1}},
            {'$project': {
                'date': '$session_date',
                'percentage': f'$behaviors.{behavior_type}.percent'
            }}
        ]
        
        results = list(reports_collection.aggregate(pipeline))
        
        trends = []
        for result in results:
            trends.append({
                'date': result['date'].isoformat(),
                'percentage': result.get('percentage', 0)
            })
        
        return trends
        
    except Exception as e:
        print(f"⚠ Error getting trends: {e}")
        return []


def get_class_summary():
    """
    Get class-wide summary statistics
    """
    try:
        pipeline = [
            {'$group': {
                '_id': '$student_id',
                'avg_engagement': {'$avg': '$engagement_score'},
                'total_sessions': {'$sum': 1}
            }},
            {'$sort': {'avg_engagement': -1}}
        ]
        
        results = list(reports_collection.aggregate(pipeline))
        
        summary = []
        for result in results:
            summary.append({
                'student_id': result['_id'],
                'avg_engagement': result['avg_engagement'],
                'total_sessions': result['total_sessions']
            })
        
        return summary
        
    except Exception as e:
        print(f"⚠ Error getting class summary: {e}")
        return []


# ============= Initialization =============

def init_db():
    """Initialize both DynamoDB and MongoDB"""
    print("Initializing Hybrid Database System...")
    print("=" * 60)
    
    # Initialize DynamoDB tables
    print("\n1. Setting up AWS DynamoDB...")
    init_dynamodb_tables()
    
    # Test MongoDB connection
    print("\n2. Connecting to MongoDB...")
    try:
        mongo_client.admin.command('ping')
        print(f"✓ Connected to MongoDB: {MONGODB_DB}")
        
        # Create indexes
        print("✓ Created MongoDB indexes")
        
    except Exception as e:
        print(f"⚠ MongoDB connection failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Hybrid Database initialized successfully!")
    print(f"  - DynamoDB: Students & Sessions")
    print(f"  - MongoDB: Reports, Frame Analysis, Audit Logs")
    print("=" * 60)


# ============= Testing & Sample Data =============

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Add sample students to DynamoDB
    print("\nAdding sample students to DynamoDB...")
    add_student('S001', 'John Doe', 'john@example.com', 'password123')
    add_student('S002', 'Jane Smith', 'jane@example.com', 'password123')
    add_student('S003', 'Bob Johnson', 'bob@example.com', 'password123')
    
    print("\n✓ Sample students added!")
    print("  Default password: password123")
    
    # Test query
    print("\nTesting DynamoDB query...")
    student = get_student('S001')
    if student:
        print(f"✓ Retrieved: {student['name']} ({student['student_id']})")
    
    # Test MongoDB
    print("\nTesting MongoDB...")
    all_students = get_all_students()
    print(f"✓ Total students in DynamoDB: {len(all_students)}")
    
    print("\n" + "=" * 60)
    print("Hybrid database setup complete!")
    print("=" * 60)

