CREATE DATABASE RapidKyc

USE RapidKyc

CREATE TABLE GovernmentData (
	AadharNo BIGINT Primary Key ,
	Name VARCHAR(max)
	,DateOfBirth DATE
	,Gender VARCHAR(10)
	,IssueDate DATE
	,VID BIGINT
	,Address VARCHAR(max)
	,Photo VARBINARY(max)
);


-- Insert the Yash record
INSERT INTO GovernmentData (AadharNo, Name, DateOfBirth, Gender, IssueDate, VID, Address, Photo)
VALUES (
    343492571759,
    'Yash Sharma',
    '2001-01-02',
    'Male',
    '2020-09-17',
    9118510620611649,
    'Gautam Buddha Nagar, Uttar Pradesh',
    (SELECT * FROM Openrowset (BULK 'C:\Users\Yash Sharma\Yash.jpg', Single_Blob) AS T)
);

-- Insert the Sachin record
INSERT INTO GovernmentData (AadharNo, Name, DateOfBirth, Gender, IssueDate, VID, Address, Photo)
VALUES (
    787988234580,
    'Sachin Kumar',
    '2001-07-24',
    'Male',
    '2012-04-09',
    9151545058908120,
    'Ghaziabad, Uttar Pradesh',
    (SELECT * FROM Openrowset (BULK 'C:\Users\Yash Sharma\Sachin.jpg', Single_Blob) AS T)
);

-- Insert the Mayank record
INSERT INTO GovernmentData (AadharNo, Name, DateOfBirth, Gender, IssueDate, VID, Address, Photo)
VALUES (
    531907142570,
    'Mayank Gupta',
    '1997-05-07',
    'Male',
    null,
    null,
    'Prayagraj, Uttar Pradesh',
    (SELECT * FROM Openrowset (BULK 'C:\Users\Yash Sharma\Mayank.jpg', Single_Blob) AS T)
);

-- Insert the Vikrant record
INSERT INTO GovernmentData (AadharNo, Name, DateOfBirth, Gender, IssueDate, VID, Address, Photo)
VALUES (
    623463151343,
    'Vikrant Vardhan',
    '2001-01-01',
    'Male',
    null,
    9147628064167512,
    'Bulandshahr, Uttar Pradesh',
    (SELECT * FROM Openrowset (BULK 'C:\Users\Yash Sharma\Vikrant.jpg', Single_Blob) AS T)
);



select * from GovernmentData

CREATE TABLE report (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(100) NOT NULL,
    note NVARCHAR(MAX) NOT NULL,
    created_at DATETIME DEFAULT GETDATE()
);

select * from report

CREATE TABLE Contact(
    id INT PRIMARY KEY IDENTITY(1,1),
    name NVARCHAR(100),
    email NVARCHAR(100),
    subject NVARCHAR(200),
    message NVARCHAR(MAX)
);

select * from Contact


-- Create a table to store Aadhaar details
CREATE TABLE UserUpload (
    AadhaarNumber VARCHAR(20) default null,
    DateOfBirth DATE default null,
    Name VARCHAR(100) default null,
    Gender VARCHAR(10) default null,
    UploadTime DATETIME DEFAULT GETDATE(),
	UploadedImage VARBINARY(MAX) default null 
);

select * from UserDetails

--CREATE TABLE UserDetails (
--    Name NVARCHAR(255)  NULL,
--    DateOfBirth NVARCHAR(20),
--    Gender NVARCHAR(20),
--    AadharNumber NVARCHAR(20) UNIQUE,
--    InsertedAt DATETIME DEFAULT GETDATE()
--);


CREATE TABLE Users (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    FullName VARCHAR(100) NOT NULL,
    Email VARCHAR(100) NOT NULL UNIQUE,
    Password VARCHAR(100) NOT NULL
);


select * from Users




