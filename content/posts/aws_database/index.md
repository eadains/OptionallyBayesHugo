---
title: Moving my Data to Amazon Web Services
date: 2021-08-05
tags: ["AWS", "postgresql", "data"]
summary: Lessons learned migrating my data to AWS
---

I've been using a SQLite database to store my financial data locally for a while. This meant I had to have my personal computer running to do updates, I didn't have a consistent way to access data, and ran the risk of losing my data. I decided it would be best to use Amazon Web Services (AWS) to handle data storage and updating from here on. I learned a lot along the way!

# Things that didn't work

### Amazon Aurora Serverless

I went in excited about [Amazon Aurora Serverless](https://aws.amazon.com/rds/aurora/serverless/). It seemed perfect for my needs. I don't need to make database calls very often, so it would automatically shut off and cost nothing after a period of no usage. Great! However, I eventually learned that you cannot connect to an Aurora Serverless instance like you would connect to a normal SQL database to make calls. You have to use the [Amazon Data API](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/data-api.html) through the AWS command line interface or through their various SDKs. I frankly didn't feel like putting the effort in to transition all of my current SQL calls and data handling to this, so I abandoned it.

I ended up using a normal [Relational Database Service](https://aws.amazon.com/rds/) instance. I'm using the smallest instance, which only costs about 15 dollars a month and is suitable for my needs.

### AWS Lambda

Again, [Lambda](https://aws.amazon.com/lambda/) seemed perfect for my needs. It would run my python code to update my database on a schedule, charge me for that usage, then shut off and cost nothing the rest of the time. However, I found the documentation to be difficult at best and debugging to be challenging.

You have to upload your all of your package dependencies with your code in a zip file that ends up being 100's of megabytes. The size means that you can't use Amazon's cloud IDE to update your code, so you have to re-upload it every time you need to change it, and it's not fast. This made fixing bugs very tedious. You also have to ensure your zip file has certain read/write privileges before you upload it, something that I only found out via Stack Overflow after having errors.

Next up was scheduling. I need this to run nightly. You have to go through [Amazon CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/RunLambdaSchedule.html) to do this. Ugh.

I ended up using an [EC2](https://aws.amazon.com/ec2/) instance here. The smallest one is part of the free tier, so I don't even have to pay for it! 

# Setup

I have found the AWS documentation to be generally hard to get through. Much of it seems to assume that you have background in AWS already, so it's confusing to start from scratch. I want to document what I did to get this all working.

### VPC

AWS creates a virtual network for you to connect all of your instances together. It's also key to setup properly so you can access your instances from a computer connected via the internet. The easiest way to start is by using the automatic wizard from the dashboard page on the VPC console:

{{< figure src="./vpc_1.png" align=center >}}

Then you can select the option for a single public subnet:

{{< figure src="./vpc_2.png" align=center >}}

In the next screen everything can remain the default, and a name for the VPC can be entered. This process automatically creates a VPC, an internet gateway, and a subnet with that gateway attached. We do want to create another subnet under our VPC that is in a different availability zone. This is relevant for the database setup because Amazon puts the backup in a different zone than the database itself. Going to the subnets tab there should be a single subnet under the VPC you created, make note of its availability zone ID. Then you can create a new subnet under that same VPC. The IP block will need to be different. For instance the default subnet will be something like 10.0.0.0/24 so this new subnet will need to be 10.0.1.0/24. Then select an availability zone that is different than the default subnet.

Next up is the security group that defines what connections will be accepted and from where. Create a new security group under the security heading and make the inbound rules look like this:

{{< figure src="./security_group_setup_1.png" align=center >}}

The two top rules are so other instances in your subnets can connect to your database. The third can be set to accept connections from your personal computers IP by selecting "My IP" in the source box. The fourth has a type of SSH, again from your own IP, this allows you to connect to your EC2 instance via SSH to configure it. For outbound rules you can set destination to 0.0.0.0/0 and everything else to All so everything going out will be allowed.

Now the networking and security is configured!

### RDS Subnet Group

Next we have to make a subnet group for the database to use. In the RDS console, there is a subnet groups link. Create a new one, select the VPC configured earlier, and then select the two subnets. That's it!

### RDS Instance

Now moving to the database instance. Important settings to note:
- The free tier instance classes are under "Burstable classes"
- Make sure to deselect Multi-AZ deployment, this costs extra
- Select the VPC configured earlier under Connectivity, select the subnet group configured earlier, then choose the security group also configured earlier
- **Make sure that public access is set to "yes"**

Once the instance starts, on its summary page, make note of the endpoint URL and the port. This is the IP and port you'll use when connecting to the database.

### EC2 Instance

You can select a variety of machine images when creating these, I use the Ubuntu Server option. Then you can select the instance type that dictates how many resources the instance has access to. I use the free tier eligible t2.micro. On the configuration page, you can select the VPC, subnet, and other options. When you launch it, you'll be directed to download a private key file. **This is very important to keep.** This file allows you to connect to your instance via SSH.

Once launched, on the instance summary page, there is the "Public IPv4 DNS." This is the IP you'll use to connect to your instance. The SSH command to connect looks like this:
```bash
ssh -i [path to .pem file] [Instance IP address]
```

Once in, you can do whatever to get your code where it needs to be to run.

For scheduling, I use a cron job to run every night at midnight. Use `crontab -e` and put a line looking something like this:

```bash
0 0 * * * source ~/RDSDatabase/update.sh
```  
Where update.sh is whatever you need to run. Mine looks like this:

```bash
#!/bin/bash
cd ~/RDSDatabase
source venv/bin/activate
python data_update.py
```  

# Conclusion

After all the fuss of figuring this out, it has been very well worth it. My data is there and up-to-date whenever I need it. I've created some data classes to fetch and hold the data the way I need it, so I have a consistent way to access it. It all *just works*. Most importantly, it's not costing me that much money!