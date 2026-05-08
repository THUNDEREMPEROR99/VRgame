import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from interview_system import EvaluationAgent, InterviewAgent
from interview_system.models.evaluation import FinalEvaluationReport
from interview_system.config import ModelConfig, create_chat_model
from interview_system.loaders import load_questions


async def main() -> None:
    print("=" * 30)
    load_dotenv()

    questions = load_questions(Path(__file__).with_name("questions.json"))
    model = create_chat_model(
        ModelConfig(
            model=os.environ["OPENROUTER_MODEL"],
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    )
    interview = InterviewAgent(questions, model=model)
    evaluation = EvaluationAgent(model=model, session_id=interview.session_id)

    output = await interview.start()
    if output.intro_text:
        print(output.intro_text)
    print(output.next_question_text)

    while not interview.is_complete():
        answer = input("> ")
        async for event in interview.astream_answer(answer):
            for node_name, node_state in event.items():
                if node_name == "generate_character_response" and node_state.get("character_response"):
                    print(f"[stream] {node_name}: {node_state['character_response']}")
                elif node_name == "decide_stage_status" and node_state.get("stage_status"):
                    print(f"[stream] {node_name}: {node_state['stage_status']}")

        turn_output = None
        state = interview.get_state()
        if state and state.values:
            vals = state.values
            turn_output = {
                "character_response": vals.get("character_response"),
                "stage_status": vals.get("stage_status"),
                "validation_error": vals.get("validation_error"),
            }

        if turn_output and turn_output.get("character_response"):
            print(turn_output["character_response"])

        if turn_output and turn_output.get("validation_error"):
            print(f"Validation error: {turn_output['validation_error']}")

        stage_status = turn_output.get("stage_status", "continue_stage") if turn_output else "continue_stage"
        if stage_status == "ready_for_next":
            next_turn = await interview.get_next_question()
            if next_turn.is_complete:
                print(next_turn.closing_message)
            else:
                if next_turn.intro_text:
                    print(next_turn.intro_text)
                print(next_turn.next_question_text)

    summaries = await interview.wait_for_summaries()

    async for event in evaluation.astream_evaluate(interview.get_session_data(), summaries):
        for node_name, node_state in event.items():
            print(f"[eval stream] {node_name}")

    # Get the report from the already-completed graph state instead of re-invoking
    state = evaluation.graph.get_state(config=evaluation._config())
    if state.values.get("validation_error"):
        print(f"Evaluation error: {state.values['validation_error']}")
    else:
        report = FinalEvaluationReport.model_validate(state.values["final_report"])
        print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
