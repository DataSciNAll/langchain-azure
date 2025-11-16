"""Streamlit UI for the Nested Agent Travel Planner."""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional
import json
from uuid import uuid4

# Import the travel planner components
from nested_agent_travel_planner import (
    build_workflow,
    PlannerState,
    DESTINATIONS,
    _configure_otlp_tracing,
    AzureAIOpenTelemetryTracer,
    TRACER
)
from langchain_core.messages import HumanMessage, BaseMessage
from azure.monitor.opentelemetry import configure_azure_monitor
import os
from dotenv import load_dotenv

load_dotenv()

# Configure monitoring (optional)
def setup_monitoring():
    """Setup Azure monitoring if configured."""
    try:
        configure_azure_monitor(
            connection_string=os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
        )
        _configure_otlp_tracing()
    except Exception as e:
        st.warning(f"Monitoring setup failed: {e}")

def get_city_suggestions():
    """Get lists of suggested cities for dropdowns."""
    origins = ["Seattle", "New York", "San Francisco", "London", "Los Angeles", "Chicago", "Boston"]
    destinations = list(DESTINATIONS.keys()) + ["Barcelona", "Amsterdam", "Vienna", "Prague"]
    return origins, [dest.title() for dest in destinations]

def format_agent_name(agent_name: str) -> str:
    """Format agent names for display."""
    return agent_name.replace('_', ' ').title()

def main():
    st.set_page_config(
        page_title="🧭 AI Travel Planner",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧭 AI Multi-Agent Travel Planner")
    st.markdown("Plan your perfect trip with AI-powered travel specialists!")
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Trip Details")
        
        # Get city suggestions
        origins, destinations = get_city_suggestions()
        
        # Origin selection
        origin = st.selectbox(
            "🛫 Origin City",
            options=origins,
            index=0,
            help="Where will you be traveling from?"
        )
        
        # Destination selection
        destination = st.selectbox(
            "🏖️ Destination City",
            options=destinations,
            index=0,
            help="Where would you like to visit?"
        )
        
        # Date selection
        st.subheader("📅 Travel Dates")
        
        min_date = datetime.now().date()
        max_date = (datetime.now() + timedelta(days=365)).date()
        
        departure_date = st.date_input(
            "Departure Date",
            value=datetime.now().date() + timedelta(days=21),
            min_value=min_date,
            max_value=max_date
        )
        
        return_date = st.date_input(
            "Return Date",
            value=departure_date + timedelta(days=5),
            min_value=departure_date,
            max_value=max_date
        )
        
        # Number of travelers
        travelers = st.number_input(
            "👥 Number of Travelers",
            min_value=1,
            max_value=10,
            value=2
        )
        
        # Custom request
        st.subheader("✨ Special Requests")
        custom_request = st.text_area(
            "Tell us more about your trip preferences...",
            value="We'd love a boutique hotel, business-class flights and memorable activities.",
            height=100,
            help="Add any special requirements, preferences, or activities you're interested in"
        )
        
        # Plan trip button
        plan_button = st.button("🚀 Plan My Trip", type="primary", use_container_width=True)
    
    # Main content area
    if plan_button:
        # Validate dates
        if return_date <= departure_date:
            st.error("Return date must be after departure date!")
            return
            
        # Create user request
        user_request = f"We're planning a trip to {destination} from {origin} departing {departure_date.strftime('%Y-%m-%d')} and returning {return_date.strftime('%Y-%m-%d')} for {travelers} travelers. {custom_request}"
        
        # Display trip summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Route:** {origin} → {destination}")
        with col2:
            st.info(f"**Duration:** {(return_date - departure_date).days} days")
        with col3:
            st.info(f"**Travelers:** {travelers}")
        
        st.markdown("---")
        
        # Setup monitoring
        setup_monitoring()
        
        # Initialize session state for workflow tracking
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
        
        # Create initial state
        session_id = str(uuid4())
        initial_state: PlannerState = {
            "messages": [HumanMessage(content=user_request)],
            "user_request": user_request,
            "session_id": session_id,
            "origin": origin,
            "destination": destination,
            "departure": departure_date.strftime('%Y-%m-%d'),
            "return_date": return_date.strftime('%Y-%m-%d'),
            "travellers": travelers,
            "flight_summary": None,
            "hotel_summary": None,
            "activities_summary": None,
            "final_itinerary": None,
            "current_agent": "start",
        }
        
        # Build and run workflow
        workflow = build_workflow()
        app = workflow.compile()
        
        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "session_id": session_id,
                "thread_id": session_id,
            },
            "recursion_limit": 10,
        }
        
        # Create containers for real-time updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Agent workflow display
        st.subheader("🤖 AI Agents Working on Your Trip")
        
        agent_containers = {}
        agent_steps = ["coordinator", "flight_specialist", "hotel_specialist", "activity_specialist", "plan_synthesizer"]
        
        # Create containers for each agent
        for i, agent in enumerate(agent_steps):
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**{format_agent_name(agent)}**")
                with col2:
                    agent_containers[agent] = st.empty()
        
        # Run workflow with real-time updates
        try:
            step_count = 0
            total_steps = len(agent_steps)
            
            for step in app.stream(initial_state, config=config):
                node_name, node_state = next(iter(step.items()))
                step_count += 1
                
                # Update progress
                progress = min(step_count / total_steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"🔄 {format_agent_name(node_name)} is working...")
                
                # Get the latest message
                if node_state.get("messages"):
                    last_message = node_state["messages"][-1]
                    if isinstance(last_message, BaseMessage):
                        content = last_message.content
                        
                        # Display in the appropriate container
                        if node_name in agent_containers:
                            with agent_containers[node_name]:
                                # Show a preview of the agent's work
                                preview = content[:300] + "..." if len(content) > 300 else content
                                st.success(f"✅ {preview}")
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✨ Trip planning complete!")
            
            # Display final itinerary
            final_state = node_state
            final_itinerary = final_state.get("final_itinerary", "")
            
            if final_itinerary:
                st.markdown("---")
                st.subheader("🎉 Your Complete Travel Itinerary")
                
                # Create an expandable section for the full itinerary
                with st.expander("📋 View Full Itinerary", expanded=True):
                    st.markdown(final_itinerary)
                
                # Offer download option
                st.download_button(
                    label="📥 Download Itinerary",
                    data=final_itinerary,
                    file_name=f"travel_itinerary_{destination}_{departure_date}.txt",
                    mime="text/plain"
                )
                
                # Display individual summaries
                st.markdown("---")
                st.subheader("📊 Agent Summaries")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if final_state.get("flight_summary"):
                        st.info("✈️ **Flight Options**")
                        st.write(final_state["flight_summary"])
                
                with col2:
                    if final_state.get("hotel_summary"):
                        st.info("🏨 **Hotel Recommendation**")
                        st.write(final_state["hotel_summary"])
                
                with col3:
                    if final_state.get("activities_summary"):
                        st.info("🎯 **Activities & Experiences**")
                        st.write(final_state["activities_summary"])
            
        except Exception as e:
            st.error(f"An error occurred while planning your trip: {str(e)}")
            st.info("Please check your environment configuration and try again.")
    
    else:
        # Show welcome message and instructions
        st.markdown("""
        ### How it works:
        
        1. **📍 Choose your destinations** - Select where you're traveling from and to
        2. **📅 Pick your dates** - Choose your departure and return dates
        3. **👥 Add traveler details** - Specify how many people are traveling
        4. **✨ Add preferences** - Tell us about your travel style and interests
        5. **🚀 Let AI plan** - Watch as specialized AI agents create your perfect itinerary
        
        Our AI travel specialists will:
        - 🛫 Find the best flight options
        - 🏨 Recommend perfect accommodations
        - 🎯 Curate amazing activities and experiences
        - 📝 Create a polished, complete itinerary
        
        **Ready to start planning?** Fill out the form in the sidebar and click "Plan My Trip"!
        """)
        
        # Show some destination highlights
        st.markdown("---")
        st.subheader("✨ Popular Destinations")
        
        cols = st.columns(3)
        for i, (dest, info) in enumerate(list(DESTINATIONS.items())[:3]):
            with cols[i]:
                st.markdown(f"""
                **{dest.title()}, {info['country']}**
                - {info['highlights'][0]}
                - {info['highlights'][1]}
                - {info['highlights'][2]}
                """)


if __name__ == "__main__":
    main()