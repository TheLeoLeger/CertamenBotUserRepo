with tab2:
    st.title("🔍 Advanced Search")
    
    # Create sub-tabs for different difficulty levels
    adv_tab1, adv_tab2, adv_tab3 = st.tabs(["🌱 Novice", "📚 Intermediate", "🧠 Advanced"])
    
    # Store selected difficulty level in session state
    if 'difficulty_level' not in st.session_state:
        st.session_state.difficulty_level = 'novice'
    
    with adv_tab1:
        st.session_state.difficulty_level = 'novice'
        st.header("🌱 Novice Level")
        st.markdown("*Simple questions with clear, beginner-friendly explanations*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📖 Basic Questions")
            
            # Easy category selection
            easy_category = st.selectbox(
                "Choose a topic:",
                [
                    "Select a topic...",
                    "🏛️ Major Gods & Goddesses",
                    "⚡ Basic Mythology Stories", 
                    "👑 Famous Roman Emperors",
                    "🗡️ Well-known Heroes",
                    "🏛️ Important Cities",
                    "📚 Famous Literature",
                    "🔤 Common Latin Words",
                    "🎭 Basic Greek Culture",
                    "⚔️ Major Wars",
                    "🏺 Daily Life in Rome/Greece"
                ]
            )
            
            # Generate easy questions based on category
            easy_questions = {
                "🏛️ Major Gods & Goddesses": [
                    "Who is the king of the gods?",
                    "Who is the goddess of wisdom?",
                    "Who is the god of the sea?",
                    "Who is the god of war?",
                    "Who is the goddess of love?",
                    "Who is the messenger god?",
                    "Who is the god of the underworld?",
                    "Who is the goddess of the hunt?"
                ],
                "⚡ Basic Mythology Stories": [
                    "What is the story of Pandora's box?",
                    "Who was Hercules?",
                    "What happened to Icarus?",
                    "Who was Medusa?",
                    "What is the Trojan Horse story?",
                    "Who was Perseus?",
                    "What is the story of King Midas?",
                    "Who was Theseus?"
                ],
                "👑 Famous Roman Emperors": [
                    "Who was Julius Caesar?",
                    "Who was Augustus?",
                    "Who was Nero?",
                    "Who was Trajan?",
                    "Who was Marcus Aurelius?",
                    "Who was Constantine?",
                    "Who was Hadrian?",
                    "Who was Caligula?"
                ],
                "🗡️ Well-known Heroes": [
                    "Who was Achilles?",
                    "Who was Odysseus?",
                    "Who was Aeneas?",
                    "Who was Hector?",
                    "Who was Jason?",
                    "Who was Bellerophon?",
                    "Who was Cadmus?",
                    "Who was Orpheus?"
                ],
                "🔤 Common Latin Words": [
                    "What does 'carpe diem' mean?",
                    "What does 'et cetera' mean?",
                    "What does 'veni, vidi, vici' mean?",
                    "What does 'alma mater' mean?",
                    "What does 'ad hoc' mean?",
                    "What does 'per se' mean?",
                    "What does 'vice versa' mean?",
                    "What does 'status quo' mean?"
                ]
            }
            
            if easy_category != "Select a topic..." and easy_category in easy_questions:
                st.write("**Sample questions:**")
                for i, question in enumerate(easy_questions[easy_category][:6], 1):
                    if st.button(f"{i}. {question}", key=f"easy_{i}_{easy_category}"):
                        st.session_state.level_query = question
                        st.session_state.query_level = 'novice'
            
            # Custom novice question
            st.subheader("💬 Ask Your Own Simple Question")
            novice_custom = st.text_input(
                "Ask a basic question:",
                placeholder="e.g., Who is Jupiter? What is the Colosseum?",
                key="novice_input"
            )
            
            if st.button("🔍 Search (Novice Level)", key="novice_search"):
                if novice_custom:
                    st.session_state.level_query = novice_custom
                    st.session_state.query_level = 'novice'
        
        with col2:
            st.subheader("🎯 Novice Features")
            st.info("""
            **At Novice Level, you get:**
            • Simple, clear explanations
            • Basic facts and definitions
            • Easy-to-understand examples
            • No complex historical context
            • Visual aids when helpful
            """)
            
            if st.button("🎲 Random Easy Question"):
                import random
                easy_randoms = [
                    "Who is Zeus?", "Who is Venus?", "What is Mount Olympus?",
                    "Who was Cleopatra?", "What is the Pantheon?", "Who is Apollo?",
                    "What does SPQR stand for?", "Who was Homer?", "What is Latin?",
                    "Who is Diana?", "What is a gladiator?", "Who is Neptune?"
                ]
                st.session_state.level_query = random.choice(easy_randoms)
                st.session_state.query_level = 'novice'

    with adv_tab2:
        st.session_state.difficulty_level = 'intermediate'
        st.header("📚 Intermediate Level")
        st.markdown("*More complex questions with detailed explanations and context*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Intermediate Questions")
            
            intermediate_category = st.selectbox(
                "Choose a topic area:",
                [
                    "Select a topic...",
                    "🏛️ Divine Relationships & Family Trees",
                    "⚔️ Military Tactics & Strategies",
                    "🏛️ Political Systems & Governance",
                    "📚 Literary Analysis & Themes",
                    "🎭 Religious Practices & Festivals",
                    "🏺 Social Classes & Daily Life",
                    "🗺️ Geographic & Cultural Connections",
                    "💰 Economic & Trade Systems",
                    "🎨 Art & Architectural Styles",
                    "📜 Historical Cause & Effect"
                ]
            )
            
            intermediate_questions = {
                "🏛️ Divine Relationships & Family Trees": [
                    "How are Zeus and Hera related, and what conflicts arose from their marriage?",
                    "What is the relationship between Greek and Roman versions of the same gods?",
                    "How did the Titans relate to the Olympian gods?",
                    "What role did divine intervention play in the Trojan War?",
                    "How did Roman gods differ in character from their Greek counterparts?",
                    "What were the major love affairs of Jupiter and their consequences?"
                ],
                "⚔️ Military Tactics & Strategies": [
                    "How did the Roman legion system evolve over time?",
                    "What were the key differences between Greek and Roman military tactics?",
                    "How did Hannibal's strategy work during the Punic Wars?",
                    "What role did siegecraft play in Roman conquests?",
                    "How did naval warfare develop in the ancient Mediterranean?",
                    "What were the advantages of the Roman military road system?"
                ],
                "🏛️ Political Systems & Governance": [
                    "How did the Roman Republic's government structure work?",
                    "What were the differences between Athenian and Spartan government?",
                    "How did the transition from Republic to Empire affect Roman society?",
                    "What role did the Senate play in Roman politics?",
                    "How did Roman citizenship evolve and expand?",
                    "What were the causes and effects of the Gracchi brothers' reforms?"
                ],
                "📚 Literary Analysis & Themes": [
                    "What are the major themes in Virgil's Aeneid?",
                    "How does Homer's Odyssey reflect Greek values?",
                    "What literary techniques did Ovid use in his Metamorphoses?",
                    "How do Greek tragedies explore the concept of fate versus free will?",
                    "What role does honor play in Roman epic poetry?",
                    "How did Roman poets adapt Greek literary forms?"
                ]
            }
            
            if intermediate_category != "Select a topic..." and intermediate_category in intermediate_questions:
                st.write("**Sample questions:**")
                for i, question in enumerate(intermediate_questions[intermediate_category][:5], 1):
                    if st.button(f"{i}. {question}", key=f"inter_{i}_{intermediate_category}"):
                        st.session_state.level_query = question
                        st.session_state.query_level = 'intermediate'
            
            # Custom intermediate question
            st.subheader("💬 Ask Your Own Detailed Question")
            intermediate_custom = st.text_area(
                "Ask a more complex question:",
                placeholder="e.g., How did the Roman Republic's political system influence later governments? What were the causes of the fall of Troy?",
                height=100,
                key="intermediate_input"
            )
            
            if st.button("🔍 Search (Intermediate Level)", key="intermediate_search"):
                if intermediate_custom:
                    st.session_state.level_query = intermediate_custom
                    st.session_state.query_level = 'intermediate'
        
        with col2:
            st.subheader("🎯 Intermediate Features")
            st.info("""
            **At Intermediate Level, you get:**
            • Detailed explanations with context
            • Historical background information
            • Cause-and-effect relationships
            • Multiple perspectives on events
            • Connections between topics
            • Primary source references
            """)
            
            st.subheader("🔄 Comparison Tools")
            comp_topic1 = st.text_input("Compare topic 1:", placeholder="e.g., Athens")
            comp_topic2 = st.text_input("Compare topic 2:", placeholder="e.g., Sparta")
            
            if st.button("⚖️ Compare (Intermediate)"):
                if comp_topic1 and comp_topic2:
                    st.session_state.level_query = f"Compare and contrast {comp_topic1} and {comp_topic2}, including their similarities, differences, and historical significance"
                    st.session_state.query_level = 'intermediate'

    with adv_tab3:
        st.session_state.difficulty_level = 'advanced'
        st.header("🧠 Advanced Level")
        st.markdown("*Complex, scholarly questions with comprehensive analysis*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎓 Advanced Research Questions")
            
            advanced_category = st.selectbox(
                "Choose a specialized area:",
                [
                    "Select a topic...",
                    "🔬 Textual Criticism & Manuscript Traditions",
                    "🏛️ Prosopography & Social Networks",
                    "📜 Epigraphy & Archaeological Evidence",
                    "🎭 Syncretism & Religious Evolution",
                    "💰 Economic Systems & Trade Networks",
                    "🗣️ Linguistics & Language Evolution",
                    "🎨 Iconography & Symbolic Systems",
                    "⚖️ Legal Systems & Jurisprudence",
                    "🌍 Cultural Exchange & Hellenization",
                    "📊 Historiography & Source Analysis"
                ]
            )
            
            advanced_questions = {
                "🔬 Textual Criticism & Manuscript Traditions": [
                    "How do the manuscript traditions of Homer's epics reflect their oral origins?",
                    "What can textual variants in Virgil's Aeneid tell us about ancient editorial practices?",
                    "How did the transmission of Ovid's works influence medieval European literature?",
                    "What role did Byzantine scholars play in preserving classical texts?",
                    "How do papyrus discoveries change our understanding of ancient literary practices?"
                ],
                "🏛️ Prosopography & Social Networks": [
                    "How did senatorial families maintain power across multiple generations in the late Republic?",
                    "What role did marriage alliances play in Julio-Claudian dynastic politics?",
                    "How did the equestrian order's economic activities influence imperial policy?",
                    "What can we learn from the social networks of Greek intellectuals in the Hellenistic period?",
                    "How did client-patron relationships structure Roman provincial administration?"
                ],
                "🎭 Syncretism & Religious Evolution": [
                    "How did the cult of Isis adapt to different cultural contexts across the Mediterranean?",
                    "What factors influenced the development of Greco-Buddhist art in ancient Gandhara?",
                    "How did mystery religions compete with traditional Roman state religion?",
                    "What role did solar theology play in late Roman imperial ideology?",
                    "How did Christian apologetics engage with classical philosophical traditions?"
                ],
                "🗣️ Linguistics & Language Evolution": [
                    "How did the development of Koine Greek affect literary style in the Hellenistic period?",
                    "What can the Vulgar Latin inscriptions tell us about spoken language evolution?",
                    "How did bilingualism function in the Roman provinces?",
                    "What role did Greek loan words play in the development of Latin technical vocabulary?",
                    "How did the Second Sophistic movement influence literary Greek style?"
                ],
                "⚖️ Legal Systems & Jurisprudence": [
                    "How did the development of Roman law influence provincial legal systems?",
                    "What role did legal rhetoric play in the education of Roman elites?",
                    "How did the Digest of Justinian preserve and transform classical legal principles?",
                    "What can legal papyri tell us about provincial administration in Roman Egypt?",
                    "How did concepts of citizenship evolve in Roman legal thought?"
                ]
            }
            
            if advanced_category != "Select a topic..." and advanced_category in advanced_questions:
                st.write("**Sample research questions:**")
                for i, question in enumerate(advanced_questions[advanced_category][:4], 1):
                    if st.button(f"{i}. {question}", key=f"adv_{i}_{advanced_category}"):
                        st.session_state.level_query = question
                        st.session_state.query_level = 'advanced'
            
            # Very specific advanced questions
            st.subheader("🔬 Expert-Level Queries")
            expert_examples = [
                "Who was Terpander and what was his role in the development of Greek music theory?",
                "What is the significance of the Dionysiac technitai in Hellenistic theater?",
                "How did the lex Iulia de Maritandis Ordinibus affect Roman demographic patterns?",
                "What was the role of the grammaticus in Roman educational curriculum?",
                "Who was Menecrates of Xanthos and what was his historical methodology?",
                "What is the significance of the Fasti Capitolini for Roman chronology?",
                "How did the cursus honorum evolve during the Principate?",
                "What was the role of the scriba in Roman administrative hierarchy?"
            ]
            
            selected_expert = st.selectbox("Or choose an expert question:", ["Select..."] + expert_examples)
            if selected_expert != "Select...":
                if st.button("🎯 Explore Expert Question"):
                    st.session_state.level_query = selected_expert
                    st.session_state.query_level = 'advanced'
            
            # Custom advanced question
            st.subheader("💬 Your Research Question")
            advanced_custom = st.text_area(
                "Pose a scholarly research question:",
                placeholder="e.g., How did the philosophical schools of the Hellenistic period influence Roman intellectual development, and what evidence do we have for cross-cultural philosophical dialogue?",
                height=120,
                key="advanced_input"
            )
            
            if st.button("🔍 Research (Advanced Level)", key="advanced_search"):
                if advanced_custom:
                    st.session_state.level_query = advanced_custom
                    st.session_state.query_level = 'advanced'
        
        with col2:
            st.subheader("🎯 Advanced Features")
            st.info("""
            **At Advanced Level, you get:**
            • Comprehensive scholarly analysis
            • Multiple source perspectives
            • Historiographical discussion
            • Primary source citations
            • Modern scholarly debates
            • Cross-cultural comparisons
            • Methodological considerations
            """)
            
            st.subheader("🕸️ Relationship Mapping")
            central_figure = st.text_input(
                "🎭 Central Figure/Concept:",
                placeholder="e.g., Cicero, Stoicism, Augustan Poetry"
            )
            
            if st.button("🗺️ Generate Network Analysis"):
                if central_figure:
                    st.session_state.level_query = f"Provide a comprehensive network analysis of {central_figure}, including intellectual influences, social connections, cultural impact, and historical significance with detailed source analysis"
                    st.session_state.query_level = 'advanced'
            
            st.subheader("📊 Specialized Tools")
            if st.button("📜 Manuscript Analysis"):
                st.session_state.level_query = "Discuss the manuscript tradition and textual criticism of a major classical work"
                st.session_state.query_level = 'advanced'
            
            if st.button("🏛️ Prosopographical Study"):
                st.session_state.level_query = "Analyze the family connections and social networks of Roman senatorial families"
                st.session_state.query_level = 'advanced'
    
    # Process and display results based on selected level
    if "level_query" in st.session_state and st.session_state.level_query:
        st.divider()
        query = st.session_state.level_query
        level = st.session_state.get('query_level', 'novice')
        
        # Create level-appropriate prompts
        level_prompts = {
            'novice': """
            Provide a simple, clear explanation suitable for beginners. Include:
            - Basic facts and definitions
            - Simple, easy-to-understand language
            - Key points without overwhelming detail
            - Visual descriptions when helpful
            - Clear, direct answers
            Keep the explanation concise but complete for someone new to the topic.
            """,
            'intermediate': """
            Provide a detailed explanation with historical context. Include:
            - Comprehensive background information
            - Cause-and-effect relationships
            - Historical significance and impact
            - Multiple perspectives when relevant
            - Connections to related topics
            - Some primary source references
            Use scholarly language but remain accessible.
            """,
            'advanced': """
            Provide a comprehensive scholarly analysis. Include:
            - Detailed historiographical discussion
            - Multiple primary and secondary sources
            - Modern scholarly debates and interpretations
            - Cross-cultural and comparative analysis
            - Methodological considerations
            - Textual criticism where relevant
            - Full academic context and implications
            Use advanced scholarly language and provide extensive detail.
            """
        }
        
        enhanced_query = f"{query}\n\n{level_prompts[level]}"
        
        st.subheader(f"🔍 {level.title()} Level Results")
        st.write(f"**Query:** {query}")
        st.write(f"**Explanation Level:** {level.title()}")
        
        with st.spinner(f"🧠 Preparing {level} level analysis..."):
            try:
                result = qa({"query": enhanced_query})
                
                answer = result["result"]
                sources = result.get("source_documents", [])
                
                # Display results with level-appropriate formatting
                if level == 'novice':
                    st.markdown("### 📖 Simple Explanation")
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Basic Sources", expanded=False):
                            for i, doc in enumerate(sources[:3], 1):  # Show fewer sources for novices
                                source_name = doc.metadata.get('source', 'Unknown')
                                st.write(f"**{i}.** {source_name}")
                
                elif level == 'intermediate':
                    st.markdown("### 📚 Detailed Analysis")
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander(f"📚 Sources & References ({len(sources)} sources)", expanded=False):
                            for i, doc in enumerate(sources[:6], 1):  # Show moderate number of sources
                                source_name = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'N/A')
                                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.markdown(f"""
                                **Source {i}:** {source_name} (Page: {page})  
                                *Preview:* {content_preview}
                                """)
                
                else:  # advanced
                    st.markdown("### 🎓 Scholarly Analysis")
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("### 📚 Complete Source Analysis")
                        
                        # Advanced source categorization
                        primary_sources = []
                        secondary_sources = []
                        other_sources = []
                        
                        for doc in sources:
                            source_name = doc.metadata.get('source', 'Unknown').lower()
                            if any(term in source_name for term in ['inscription', 'papyrus', 'manuscript', 'ancient']):
                                primary_sources.append(doc)
                            elif any(term in source_name for term in ['modern', 'scholar', 'analysis', 'study']):
                                secondary_sources.append(doc)
                            else:
                                other_sources.append(doc)
                        
                        if primary_sources:
                            with st.expander(f"📜 Primary Sources ({len(primary_sources)})", expanded=True):
                                for i, doc in enumerate(primary_sources, 1):
                                    source_name = doc.metadata.get('source', 'Unknown')
                                    page = doc.metadata.get('page', 'N/A')
                                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    st.markdown(f"""
                                    **{i}. {source_name}** (Page: {page})  
                                    {content}
                                    """)
                                    st.divider()
                        
                        if secondary_sources:
                            with st.expander(f"🎓 Secondary Sources ({len(secondary_sources)})", expanded=False):
                                for i, doc in enumerate(secondary_sources, 1):
                                    source_name = doc.metadata.get('source', 'Unknown')
                                    page = doc.metadata.get('page', 'N/A')
                                    content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                                    st.markdown(f"""
                                    **{i}. {source_name}** (Page: {page})  
                                    {content}
                                    """)
                                    st.divider()
                        
                        if other_sources:
                            with st.expander(f"📚 Additional Sources ({len(other_sources)})", expanded=False):
                                for i, doc in enumerate(other_sources, 1):
                                    source_name = doc.metadata.get('source', 'Unknown')
                                    page = doc.metadata.get('page', 'N/A')
                                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.markdown(f"""
                                    **{i}. {source_name}** (Page: {page})  
                                    {content}
                                    """)
                
                # Clear the query after displaying results
                del st.session_state.level_query
                if 'query_level' in st.session_state:
                    del st.session_state.query_level
                
            except Exception as e:
                st.error(f"❌ Error in {level} level analysis: {e}")
    
    # Tips section
    with st.expander("💡 How the Difficulty Levels Work", expanded=False):
        st.markdown("""
        ### 🎯 Understanding the Three Levels
        
        **🌱 Novice Level**
        - **Questions:** Basic identification and simple "who/what/where" questions
        - **Examples:** "Who is Zeus?" "What is the Colosseum?" "What does SPQR mean?"
        - **Answers:** Simple, clear explanations with basic facts
        - **Perfect for:** Beginners, quick facts, study review
        
        **📚 Intermediate Level**  
        - **Questions:** More complex "how/why" questions requiring analysis
        - **Examples:** "How did Roman government work?" "Why did the Trojan War start?"
        - **Answers:** Detailed explanations with historical context and connections
        - **Perfect for:** Students preparing for competitions, deeper understanding
        
        **🧠 Advanced Level**
        - **Questions:** Scholarly research questions with multiple variables
        - **Examples:** "Who was the lyre teacher of Nero?" "How did Hellenistic philosophy influence Roman jurisprudence?"
        - **Answers:** Comprehensive analysis with primary sources and scholarly debate
        - **Perfect for:** Advanced students, researchers, complex certamen questions
        
        ### 🔍 Same Information, Different Depth
        
        All levels access the **same complete knowledge base** - the difference is in explanation complexity:
        - **Novice:** Gets the essential facts clearly explained
        - **Intermediate:** Gets context, background, and connections  
        - **Advanced:** Gets full scholarly treatment with sources and analysis
        
        Choose the level that matches your needs and understanding!
        """)
