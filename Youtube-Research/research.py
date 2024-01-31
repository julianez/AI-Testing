import requests
from youtubesearchpython import VideosSearch
import re  # For email extraction
import time  # For rate limiting

# Authentication for YouTube Data API (replace with your API key)
api_key = ""

# Function to search for channels meeting the criteria
def find_channels(topics):
    channels = []
    for topic in topics:
        search = VideosSearch(topic, limit=500)
        results = search.result()["result"]
        
        for video in results:
            channel_id = video["channel"]["id"]
            channel_title = video["channel"]["name"]            
            subscriber_count = get_subscriber_count(channel_id)
            view_count_text = video["viewCount"]["short"]
            view_count = parse_view_count (view_count_text)

            if subscriber_count > 5000 and view_count > 10000:
                channels.append({
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "subscriber_count": subscriber_count,
                    "video_url": video["link"],
                })
    return channels

def get_subscriber_count(channel_id):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={channel_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["items"]:
        subscriber_count = data["items"][0]["statistics"]["subscriberCount"]
        return int(subscriber_count)
    else:
        return None

def parse_view_count(view_str):
    """
    Parses a view count string like "544K views" or "432M views" into an integer.

    Args:
    view_str: A string representing the view count.

    Returns:
    The view count as an integer.
    """
    match = re.match(r"(\d+)([KMGT]?)", view_str)
    if not match:
        raise ValueError(f"Invalid view count string: {view_str}")
    count, unit = match.groups()
    count = int(count)
    multiplier = {
      "K": 1000,
      "M": 1000000,
      "G": 1000000000,
      "T": 1000000000000,
    }.get(unit, 1)  # Default multiplier is 1 for missing unit
    
    return count * multiplier
    
# Function to fetch channel details
def get_channel_details(channel_id):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={channel_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["items"]:
        channel_data = data["items"][0]["snippet"]
        return channel_data
    else:
        return None

# Function to extract email from channel description (basic approach)
def extract_email(description):
    email_regex = r"[\w\.-]+@[\w\.-]+\.\w+"
    matches = re.findall(email_regex, description, flags=re.IGNORECASE)
    return matches[0] if matches else None

# Function to find social media links (potential URLs in description)
def find_social_links(description):
    social_links = {}
    for link in re.findall(r"https?://\S+", description):
        if "linkedin" in link:
            social_links["linkedin"] = link
        elif "twitter" in link:
            social_links["twitter"] = link
        elif "facebook" in link:
            social_links["facebook"] = link
        elif "instagram" in link:
            social_links["instagram"] = link
    return social_links

# Main function to execute the process
def main():
    topics = ["langchain", "llamaindex", "crewai", "autogen", "open interpreter"]
    influencers = []

    channels = find_channels(topics)
    # Store unique channel IDs to avoid duplicates
    unique_channel_ids = set()  # Use a set for efficient lookups
    filtered_channels = []

    for channel in channels:
        channel_id = channel["channel_id"]  # Assuming "channel_id" holds the unique identifier
        if channel_id not in unique_channel_ids:
            unique_channel_ids.add(channel_id)  # Add ID to the set of seen IDs
            filtered_channels.append(channel)  # Add channel to the filtered list

    for channel in filtered_channels:
        channel_data = get_channel_details(channel["channel_id"])
        print (channel_data)
        if channel_data:
            email = extract_email(channel_data["description"])
            social_links = find_social_links(channel_data["description"])
            website = channel_data.get("websiteUrl")

            influencer = {
                "channelTitle":channel_data["title"],
                "channel_url": f"https://www.youtube.com/channel/{channel['channel_id']}",
                "email": email,
                "example_video_url": channel["video_url"],
                "website": website,
                "social_media": social_links,
            }
            influencers.append(influencer)

            # Rate limiting (adjust as needed)
            time.sleep(1)

        if len(influencers) >= 200:
            break

    # Print or store the influencers list
    for influencer in influencers:
        print(f"**Channel:** {influencer['channelTitle']}")
        print(f"- Channel URL: {influencer['channel_url']}")
        print(f"- Email: {influencer['email']}")
        print(f"- Example Video URL: {influencer['example_video_url']}")  # Assuming you've stored an example video URL
        print(f"- Website: {influencer['website']}")
        print(f"- LinkedIn: {influencer.get('linkedin', 'Not available')}")
        print(f"- Twitter: {influencer.get('twitter', 'Not available')}")
        print(f"- Social: {influencer['social_media']}")
        print("----------------------------------")
        

if __name__ == "__main__":
    main()
