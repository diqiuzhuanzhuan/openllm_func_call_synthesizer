# MIT License
#
# Copyright (c) 2025, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
UGREEN Media Server MCP
A FastMCP server implementing photo, music, video, and system control functions
"""

import datetime
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

# Initialize FastMCP server
mcp = FastMCP("UGREEN Media Server")


@mcp.tool
def create_album(
    album_name: Annotated[str, Field(default=..., description="The name of the album to be created.")],
    search_query: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "search keyword or filter used to find photos "
                '(e.g., "beach", "family", "2024 vacation"). The album will include the'
                " photos that match this query."
            ),
            examples=["beach", "family", "2024 vacation"],
        ),
    ],
    album_type: Annotated[
        Literal["normal"],
        #                "baby", "face", "condition", "object"],
        Field(
            default="normal",
            description="The type of album to create. Valid options: normal.",
            examples=["normal"],
        ),
    ],
) -> dict[str, Any]:
    """
    Create a new photo album, optionally using search results from the photo library.
    """

    if album_type not in ["normal"]:
        result = {"error": "Invalid album_type."}
        return result

    # Simulate album creation
    message_suffix = f" based on query '{search_query}'" if search_query else ""
    result = {
        "status": "success",
        "message": f"Album '{album_name}' created successfully{message_suffix}",
        "album_name": album_name,
        "album_type": album_type,
        "search_query": search_query,
    }
    print(result)
    return result


@mcp.tool()
def search_photos(
    keyword: Annotated[
        str,
        Field(
            default=...,
            description="""The search keyword used to find photos or images.
            This can be descriptive text or a file name.
            """,
            examples=["photos taken last August", "dog on the grass"],
        ),
    ],
) -> dict[str, Any]:
    """
    Search for photos or images.
    """
    # Simulate photo search
    result = {
        "status": "success",
        "keyword": keyword,
        "results": [
            {"filename": f"photo1_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-15"},
            {"filename": f"photo2_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-16"},
            {"filename": f"photo3_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-17"},
        ],
        "count": 3,
    }
    print(result)
    return result


@mcp.tool()
def get_album_list(
    album_type: Annotated[
        Literal["normal", "face", "baby", "condition", "object"],
        Field(
            default=None,
            description="""The type of album to retrieve. Valid options:
            - `normal`: regular albums
            - `face`: people albums
            - `baby`: baby albums
            - `condition`: conditional albums
            - `object`: object-recognition albums
            """,
        ),
    ],
    keyword: Annotated[
        str,
        Field(
            default=None,
            description="An optional search keyword used to filter albums by name or content.",
            examples=["vacation", "family", "2024"],
        ),
    ],
) -> dict[str, Any]:
    """
    Retrieve a list of photo albums.
    Either `album_type` or `keyword` must be provided; both are optional but at
    least one is required.
    """
    if not album_type and not keyword:
        result = {"error": "Either album_type or keyword must be provided."}
        print(result)
        return result

    valid_types = ["normal", "face", "baby", "condition", "object"]
    if album_type not in valid_types:
        result = {"error": f"Invalid album_type. Must be one of: {valid_types}"}
        print(result)
        return result

    # Simulate album list retrieval
    album_lists = {
        "normal": ["Vacation 2024", "Family Photos", "Work Events"],
        "face": ["John Doe", "Jane Smith", "Mike Johnson"],
        "baby": ["Baby's First Year", "Milestones"],
        "condition": ["Sunny Days", "Indoor Photos", "Night Shots"],
        "object": ["Cars", "Animals", "Food", "Buildings"],
    }
    result = {
        "status": "success",
        "message": f"Retrieving album list filtered by type: {album_type} and keyword: {keyword}",
        "album_type": album_type,
        "keyword": keyword,
        "albums": album_lists.get(album_type, []),
    }
    print(result)
    return result


# --- Define the Enums ---
MusicSource = Literal["recent", "favorites", "playlist"]
PlayMode = Literal["normal", "shuffle", "repeat_one", "repeat_all"]


@mcp.tool(name="music_play_control")
def music_play_control(
    title: Annotated[
        str | None,  # Explicitly mark as Optional using Optional[str]
        Field(
            description=(
                "The search query for the music content (song, album, artist, or specific playlist name)."
                "REQUIRED unless 'source' is specified."
            ),
            examples=["Hotel California", "Taylor Swift", "Workout Mix"],
            default=None,  # Explicitly set default to None
        ),
    ],
    source: Annotated[
        MusicSource | None,  # Explicitly mark as Optional
        Field(
            default=None,
            description="""The content source to play from. Valid options:
            - `recent`: The recently played songs list.
            - `favorites`: The user's liked songs.
            - `playlist`: Songs within the user's saved playlists.

            Only specify this argument when the user explicitly mentions playing from these predefined lists.
            """,
        ),
    ],
    play_mode: Annotated[
        PlayMode,
        Field(
            description="""Playback mode. Valid values:
            - `normal`: Sequential playback through the list.
            - `shuffle`: Random playback order.
            - `repeat_one`: Repeat the current track indefinitely (Repeat One).
            - `repeat_all`: Repeat all tracks in the current list (Repeat All).
            """,
            default="normal",  # Pydantic default for schema clarity
        ),
    ] = "normal",  # Python default for function clarity
) -> dict[str, Any]:
    """
    Music control tool for initiating playback of specific content, albums, artists, or predefined sources.

    ***USAGE REQUIREMENT***
    Either 'title' (a search query) OR 'source' (a predefined list) must be provided for playback to start.
    """

    if not title and not source:
        result = {"error": "Either title (search query) or source (predefined list) must be provided."}
        print(result)
        return result

    valid_sources = ["recent", "favorites", "playlist"]
    valid_modes = ["normal", "shuffle", "repeat_one", "repeat_all"]

    if source and source not in valid_sources:
        result = {"error": f"Invalid source. Must be one of: {valid_sources}"}
        print(result)
        return result

    if play_mode not in valid_modes:
        result = {"error": f"Invalid play_mode. Must be one of: {valid_modes}"}
        print(result)
        return result

    response = {"status": "success", "action": "play", "play_mode": play_mode}

    if title:
        response["title"] = title
        response["message"] = f"Playing '{title}' in {play_mode} mode"
    elif source:
        response["source"] = source
        response["message"] = f"Playing from {source} in {play_mode} mode"
    print(response)
    return response


@mcp.tool()
def music_settings_control(
    auto_stop_time: Annotated[
        float,
        Field(
            default=...,
            description=(
                "The sleep-timer duration in minutes."
                "For example, setting this to 15 will stop playback after 15 minutes."
            ),
            examples=[15, 30, 60],
        ),
    ],
) -> dict[str, Any]:
    """
    Control music app settings. The measurement unit for time values is minutes.
    """
    if auto_stop_time <= 0:
        result = {"error": "auto_stop_time must be greater than 0"}
        print(result)
        return result

    result = {
        "status": "success",
        "message": f"Sleep timer set to {auto_stop_time} minutes",
        "auto_stop_time": auto_stop_time,
    }
    print(result)
    return result


@mcp.tool()
def music_search_control(
    keyword: Annotated[
        str,
        Field(
            default=...,
            description="""The search keyword used to find music content,
            such as song name, artist name, palylist name, or album title.
            """,
            examples=["Love Story", "Taylor Swift", "Fearless", "my playlist"],
        ),
    ],
) -> dict[str, Any]:
    """
    Music search tool: search for songs, albums, artists, or playlists based on keywords.
    """

    result = {
        "status": "success",
        "search_keyword": keyword,
        "results": [f"Song 1 by {keyword}", f"Album 2 related to {keyword}"],
    }
    print(result)
    return result


# Video Control Tools
@mcp.tool()
def video_search_control(
    title: Annotated[
        str | None,
        Field(
            description="""The name or title of the video content. Fuzzy matching is supported.
            """,
            examples=["The Shawshank Redemption", "Inception", "The Dark Knight"],
            default=None,
        ),
    ],
    source: Annotated[
        Literal["recent", "favorites", "media_library"] | None,
        Field(
            default=None,
            description="""The source of the video content. Valid options:
            - `recent`: recently played videos
            - `favorites`: liked videos
            - `media_library`: videos stored in the media library

            Only specify this argument when the user explicitly mentions recent,
            favorite, or media-library content.
            """,
        ),
    ],
    video_type: Annotated[
        Literal["tv", "movie", "collection", "all"],
        Field(
            default="all",
            description="""The content type to search. Valid values:
            - `tv`: TV series or dramas
            - `movie`: films or blockbusters
            - `collection`: movie series or collections
            - `all`: search across all types (default)

            Only specify this argument when the user explicitly mentions the type of content.
            """,
        ),
    ] = "all",  # Default value for type
) -> dict[str, Any]:
    """
    Video search tool for finding TV series, movies, and other types of video content.

    Either `title` or `source` must be provided; both are optional but at least one
    is required.
    """
    if not title and not source:
        result = {"error": "Either title or source must be provided for video search."}
        print(result)
        return result
    valid_sources = ["recent", "favorites", "media_library"]
    if source and source not in valid_sources:
        result = {"error": f"Invalid source. Must be one of: {valid_sources}"}
        print(result)
        return result

    valid_types = ["tv", "movie", "collection", "all"]
    if video_type and video_type not in valid_types:
        result = {"error": f"Invalid type. Must be one of: {valid_types}"}
        print(result)
        return result

    content_type = video_type if video_type else "all"
    # Simulate video search
    results = []
    if not video_type or video_type == "movie":
        results.append({"title": f"{title} (Movie)", "type": "movie", "year": 2023, "rating": 8.5})

    if not video_type or video_type == "tv":
        results.append({"title": f"{title} (TV Series)", "type": "tv", "seasons": 3, "rating": 9.2})

    if not video_type or video_type == "collection":
        results.append({"title": f"{title} Collection", "type": "collection", "movies": 4, "rating": 8.8})

    result = {
        "status": "success",
        "search_title": title,
        "search_type": content_type,
        "source": source,
        "results": results,
        "count": len(results),
    }
    print(result)
    return result


@mcp.tool()
def video_play_control(
    title: Annotated[
        str,
        Field(
            default=None,
            description="The name or title of the video content. Fuzzy matching is supported.",
            examples=["The Shawshank Redemption", "Inception", "The Dark Knight"],
        ),
    ],
    source: Annotated[
        Literal["recent", "favorites", "media_library"],
        Field(
            default=None,
            description="""The content source. Valid options:
            - `recent`: recently played items
            - `favorites`: liked videos
            - `media_library`: videos stored in the media library

            Only specify this argument when the user explicitly refers to recent,
            favorite, or media-library content.
            """,
        ),
    ],
    video_type: Annotated[
        Literal["tv", "movie", "collection", "all"],
        Field(
            default="all",
            description="""The content type. Valid values:
            - `tv`: TV series or dramas
            - `movie`: films or blockbusters
            - `collection`: movie series or collections
            - `all`: all types (default)
            """,
        ),
    ] = "all",  # Default value for type
) -> dict[str, Any]:
    """
    Video playback tool for playing TV series, movies, and other types of video content.

    Either `title` or `source` must be provided; both are optional but at least one
    is required.
    """
    if not title and not source:
        result = {"error": "Either title or source must be provided for video playback."}
        print(result)
        return result

    valid_types = ["tv", "movie", "collection"]
    if video_type and video_type not in valid_types:
        result = {"error": f"Invalid type. Must be one of: {valid_types}"}
        print(result)
        return result

    content_type = video_type if video_type else "all"  # Default to movie if not specified

    result = {
        "status": "success",
        "action": "play",
        "title": title,
        "type": content_type,
        "source": source,
        "message": f"Playing {content_type}: '{title}'",
    }
    print(result)
    return result


# Define the supported categories of system information
InfoCategory = Literal[
    "system",  # OS and general system info
    "device",  # Device-specific details
    "cpu",  # CPU specifications and usage details
    "ram",  # RAM usage details
    "storage",  # Disk and storage status
    "network",  # Network configuration
    "uglink",  # UGREEN Link account information
    "fan_speed",  # Device fan speed
]


@mcp.tool(name="get_system_info")
def get_system_info(
    category: Annotated[
        InfoCategory,
        Field(
            description="The specific category of system information to retrieve. Choose from: "
            "'system' (OS info, e.g., 查找系统的信息, 查找NAS的信息, NASの情報を検索する, システムの情報を検索する.), "
            "'device' (Model/Serial), "
            "'cpu' (CPU specs and usage details), "
            "'ram' (RAM usage details, e.g.,Search the information of memory), "
            "'storage' (disk and storage status, e.g., "
            "Search the information of hard drive, ハードディスクの情報を検索する), "
            "'network' (IP, MAC address, gateway, and other Network information), "
            "'uglink' (UGREEN Link, a remote access url), "
            "'fan_speed' (the speed of fan)."
        ),
    ],
) -> dict[str, Any]:
    """
    Retrieves detailed information about the system and its operational status across various predefined categories.

    Use this tool when the user asks for specs, status, usage rates, or account information related to the device.
    (For example: "Rufe die Informationen zu diesem System auf.")

    """

    # --- Mock Logic based on category ---
    current_time = datetime.datetime.now().isoformat()

    # Simulate data retrieval for different categories
    mock_data = {
        "system": {"os_version": "UGOS 5.1", "uptime": "12 days 4 hours"},
        "device": {"model": "DX4800", "serial": "UGN012345"},
        "hardware": {"cpu_model": "Intel Celeron J4125", "total_ram_gb": 8, "ram_slots": 2},
        "storage": {"total_capacity_tb": 16, "drive_count": 4, "pool_status": "Healthy"},
        "network": {"ip_address": "192.168.1.10", "gateway": "192.168.1.1"},
        "uglink": {"url": "https://ugreenlink.com/account/diqiuzhuanzhuan"},
        "hardware_rate": {"cpu_load_percent": 15, "memory_use_percent": 45, "network_utilization_mbps": 50},
        "fan_speed": {"fan_1_rpm": 1200, "fan_status": "Normal"},
    }

    retrieved_data = mock_data.get(category, {"error": "Invalid category provided."})

    # Return the retrieved data
    return {
        "status": "Success",
        "message": f"Successfully retrieved information for the '{category}' category.",
        "data": retrieved_data,
        "timestamp": current_time,
    }


# --- General and Document Tools (新增部分) ---
# @mcp.tool()
def summary_document() -> dict[str, Any]:
    """
    Summarize the content of a document or report.

    """
    result = {"status": "success", "summary": "Simulated summary for you!"}
    print(result)
    return result


# @mcp.tool()
def launch_translate() -> dict[str, Any]:
    """
    Users need to launch the translation tool in order to translate the text or documents.
    """
    result = {"status": "success"}
    print(result)
    return result


OpType = Literal["backup", "sync"]


@mcp.tool(name="start_file_backup_sync_ui")
def start_file_backup_sync_ui(
    op_type: Annotated[
        OpType,
        Field(description=("he operation mode ('backup' or 'sync') to pre-select in the launched system interface.")),
    ],
) -> dict[str, Any]:
    """
    Launches the system interface or application dedicated to file backup and synchronization management.

    ***TOOL TRIGGER RULE***
    Call this tool immediately when the user expresses a clear intent to perform a 'backup' or 'sync' operation.

    IMPORTANT: This tool does NOT require file paths. It is the first step in the workflow, \
    and paths will be configured inside the launched interface.
    """

    # Mock Logic
    if op_type not in ["backup", "sync"]:
        return {"status": "error", "message": "Invalid operation type. Must be 'backup' or 'sync'."}
    action_verb = "Backed up" if op_type == "backup" else "Synchronized"

    # Returning a clear English status message
    return {"status": "success", "message": f"{action_verb} data sucessfully."}


@mcp.tool(name="docker_search_container")
def docker_search_container(
    search_query: Annotated[
        str,
        Field(
            description=(
                "The primary keywords used to filter containers."
                "This typically matches against container name, image name, or container ID prefix."
            ),
            examples=["nginx", "db-server", "a3b1"],
        ),
    ],
) -> dict[str, Any]:
    """
    Searches and retrieves details for Docker containers matching the provided criteria.

    Use this tool when the user asks to find, list, or check the existence/status of a specific container.
    """
    # --- Mock Logic ---
    results_count = 3
    # 模拟根据参数返回数据
    return {
        "status": "success",
        "message": f"Search completed. Found {results_count} containers matching '{search_query}' (Status: running ).",
        "results": [
            {"id": "a3b1c4d", "name": f"{search_query}-web", "status": "running"},
            {"id": "d4e5f6g", "name": f"{search_query}-db", "status": "stopped"},
            {"id": "d4e5f6g", "name": f"{search_query}-db", "status": "stopped"},
        ],
    }


@mcp.tool(name="docker_search_image")
def docker_search_image(
    image_query: Annotated[
        str,
        Field(
            description="The keyword or repository name to search for (e.g., 'nginx', 'ubuntu', or 'my-registry/app').",
            examples=["nginx", "ubuntu", "my-registry/app", "redis:latest"],
        ),
    ],
) -> dict[str, Any]:
    """
    Searches for Docker images either locally on the host machine (docker images)
    or remotely on a registry (docker search).

    Use this tool when the user asks to find available images, check image versions, or discover new images to pull.
    Returns a list of matching image details (repository, tags, size, etc.).
    """
    # --- Mock Logic ---
    results_count = 3

    return {
        "status": "success",
        "message": f"Found {results_count} images matching '{image_query}'.",
        "results": [
            {"repository": image_query, "tag": "latest", "size": "150MB", "scope": "local"},
            {"repository": image_query, "tag": "1.20", "size": "145MB", "scope": "local"},
            {"repository": image_query, "tag": "1.19", "size": "143MB", "scope": "local"},
        ],
    }


# @mcp.tool(name="shutdown_nas_device")
def shutdown_nas_device(
    force: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Set to TRUE to force an immediate system shutdown, bypassing graceful processes."
                "Defaults to False (graceful shutdown)."
            ),
        ),
    ],
) -> dict[str, Any]:
    """
    Initiates an immediate power-off or shutdown sequence for the NAS device.

    Use this tool when the user explicitly requests to shut down the system.
    """

    # --- Mock Logic ---
    shutdown_type = "forced" if force else "graceful"
    return {
        "status": "success",
        "action": f"Initiated {shutdown_type} shutdown sequence for the NAS.",
        "confirmation_time": datetime.datetime.now().isoformat(),
    }


# 沿用之前的配置列表
ConfigFeature = Literal[
    "power_on_after_failure",  # 通电自动开机
    "wake_on_lan",  # 网络唤醒
    "internal_hdd_sleep",  # 硬盘休眠
    "usb_hdd_sleep",  # USB硬盘休眠
]


@mcp.tool(name="configure_nas_power_settings")
def configure_nas_power_settings(
    config_feature: Annotated[
        ConfigFeature, Field(description="The specific configuration feature to switch ON or OFF.")
    ],
    enable: Annotated[bool, Field(description="Set to TRUE to enable/open the feature, FALSE to disable/close it.")],
) -> dict[str, Any]:
    """
    Manages persistent configuration settings (sleep, wake_on_lan) on the NAS device.

    Use this tool when the user requests to enable, disable, open, or close a specific device feature.
    """

    # --- Mock Logic ---
    state = "Enabled" if enable else "Disabled"
    return {
        "status": "success",
        "config": f"Feature '{config_feature}' successfully set to '{state}'.",
        "timestamp": datetime.datetime.now().isoformat(),
    }


# Define all configurable buzzer alert events
BuzzerEvent = Literal[
    "fan_failure",  # Cooling fan failure
    "storage_abnormality",  # Storage space/SSD cache abnormality
    "system_startup",  # System startup
    "system_shutdown",  # System shutdown
    "over_temperature",  # CPU/HDD over-temperature
]


@mcp.tool(name="configure_device_buzzer_alerts")
def configure_device_buzzer_alerts(
    event_type: Annotated[
        BuzzerEvent,
        Field(
            description=(
                "The type of event that the buzzer alarm will be configured for"
                "Examples: 'fan_failure' or 'over_temperature'."
            )
        ),
    ],
    # Use boolean for clear ON/OFF state
    enable: Annotated[
        bool,
        Field(
            description=(
                "Set to TRUE to enable the buzzer alarm (turn ON),or FALSE to disable the buzzer alarm (turn OFF)."
            )
        ),
    ],
) -> dict[str, Any]:
    """
    Configures whether the device's buzzer alarm sounds when specific system events occur.

    Use this tool to enable or disable buzzer alerts for cooling fan failure, storage abnormality, \
        system power cycles, and CPU/HDD overheating conditions.
    """

    # --- Mock Logic ---
    state = "Enabled" if enable else "Disabled"
    current_time = datetime.datetime.now().isoformat()

    # Return configuration success confirmation in English
    return {
        "status": "Configuration Success",
        "message": f"Buzzer alert for event '{event_type}' successfully set to: {state}.",
        "event_configured": event_type,
        "new_state": state,
        "timestamp": current_time,
    }


# Define the set of valid fan speed control commands
FanSpeedSetting = Literal[
    "mute",  # Corresponds to 静音
    "normal",  # Corresponds to 普通
    "full_speed",  # Corresponds to 全速
    "increase",  # Corresponds to 增加
    "decrease",  # Corresponds to 减少
]


@mcp.tool(name="control_cooling_fan_speed")
def control_cooling_fan_speed(
    setting: Annotated[
        FanSpeedSetting, Field(description="The desired command or preset mode for controlling the fan speed.")
    ],
) -> dict[str, Any]:
    """
    Controls the device's cooling fan speed, allowing for preset modes or incremental adjustments.

    Use this tool when the user requests to change, adjust, increase, \
        decrease, or set a specific fan mode (e.g., mute, full speed).
    """

    # --- Mock Logic ---
    action_type = "Adjusted"
    if setting in ["mute", "normal", "full_speed"]:
        action_type = "Set to"

    current_time = datetime.datetime.now().isoformat()

    # Return confirmation of the action
    return {
        "status": "Success",
        "message": f"Cooling fan speed successfully {action_type} '{setting}'.",
        "setting_applied": setting,
        "timestamp": current_time,
    }


# Define the direction of the brightness change
BrightnessAction = Literal["increase", "decrease", "set"]


@mcp.tool(name="adjust_led_brightness")
def adjust_led_brightness(
    action: Annotated[
        BrightnessAction,
        Field(
            description=(
                "The action to perform: 'increase' (add to current), 'decrease' (subtract from current), "
                "or 'set' (set to absolute value)."
            )
        ),
    ],
    value: Annotated[
        int,
        Field(
            default=25,
            description=(
                "The value associated with the action. "
                "If action is 'set', this is the target brightness (0-100). "
                "If action is 'increase'/'decrease', this is the relative amount to change."
            ),
            ge=0,  # Greater than or equal to 0 (Pydantic constraint)
            le=100,  # Less than or equal to 100 (Pydantic constraint)
        ),
    ],
) -> dict[str, Any]:
    """
    Controls the LED light brightness. Can set a specific percentage or adjust relatively.

    Use this tool when the user says:
    - "Set brightness to 80" (action='set', value=80)
    - "Make the light brighter" (action='increase', value=25)
    - "Dim the light by 10" (action='decrease', value=10)
    """

    # --- Mock Logic ---
    current_time = datetime.datetime.now().isoformat()
    # 假设当前亮度 (仅用于演示逻辑)
    mock_current_brightness = 50
    final_brightness = 0

    # 根据 action 计算最终亮度
    if action == "set":
        final_brightness = value
    elif action == "increase":
        final_brightness = min(mock_current_brightness + value, 100)
    elif action == "decrease":
        final_brightness = max(mock_current_brightness - value, 0)

    return {
        "status": "Success",
        "message": f"LED brightness successfully updated. Action: '{action}', \
            Value: {value}. Resulting Level: {final_brightness}%",
        "data": {"action": action, "input_value": value, "final_brightness": final_brightness},
        "timestamp": current_time,
    }


# Define the supported power modes
PowerMode = Literal[
    "performance",  # Corresponds to 高性能
    "balance",  # Corresponds to 平衡
    "saving",  # Corresponds to 节能
]


@mcp.tool(name="set_power_efficiency_mode")
def set_power_efficiency_mode(
    mode: Annotated[
        PowerMode,
        Field(
            description=(
                "The desired power management profile to apply to the device. Options are: "
                "'performance' (Maximum speed and power consumption), "
                "'balance' (Optimized efficiency and speed tradeoff), or "
                "'saving' (Maximum energy saving and lower performance)."
            )
        ),
    ],
) -> dict[str, Any]:
    """
    Sets the device's overall power efficiency management profile.

    Use this tool when the user requests to optimize the system for performance, balance, or energy saving.
    """

    # --- Mock Logic ---
    current_time = datetime.datetime.now().isoformat()

    # Return confirmation of the action
    return {
        "status": "Success",
        "message": f"Power efficiency mode successfully set to '{mode}'.",
        "profile_applied": mode,
        "timestamp": current_time,
    }


@mcp.tool(name="configure_memory_compression")
def configure_memory_compression(
    enable: Annotated[
        bool, Field(description="Set to TRUE to enable the memory compression feature, or FALSE to disable it.")
    ],
) -> dict[str, Any]:
    """
    Controls the system's memory compression feature (e.g., ZRAM, ZSWAP).

    Use this tool when the user requests to turn the memory compression ON or OFF.
    """

    # --- Mock Logic ---
    state = "Enabled" if enable else "Disabled"
    current_time = datetime.datetime.now().isoformat()

    # Return confirmation of the action
    return {
        "status": "Success",
        "message": f"Memory compression feature successfully set to '{state}'.",
        "new_state": state,
        "timestamp": current_time,
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(host="0.0.0.0", port=8000, path="/mcp", transport="streamable-http")
